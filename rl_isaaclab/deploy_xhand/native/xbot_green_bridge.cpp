#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <soem_cpp/ec_master.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "ethercat_drive/hand/param_codec.h"

namespace py = pybind11;

namespace {

constexpr std::size_t kJointCount = 21;
constexpr std::size_t kSensorSlots = 33;

#pragma pack(push, 1)
struct GreenJointCommand {
    std::uint16_t joint_id;
    std::uint8_t mode;
    std::uint8_t reserved;
    std::uint16_t kp;
    std::uint16_t kd;
    std::int16_t position;
    std::int16_t speed;
    std::int16_t torque;
};

struct GreenJointState {
    std::uint16_t joint_id;
    std::int16_t position;
    std::int16_t speed;
    std::int16_t torque;
    std::uint16_t error_code;
    std::int8_t temperature;
    std::uint8_t reserved;
};

struct GreenSensor {
    std::uint8_t sensor_id;
    std::uint8_t error_code;
    std::uint32_t channel_data[11];
    std::uint32_t first_ydds_nf;
    std::uint32_t first_ydds_tf;
    std::uint16_t first_ydds_tf_dir;
    std::uint32_t second_ydds_nf;
    std::uint32_t second_ydds_tf;
    std::uint16_t second_ydds_tf_dir;
    std::uint32_t proximity;
};

struct GreenRxPdo {
    GreenJointState joints[3];
    GreenSensor sensors[3];
    std::uint8_t board_temperature;
    std::uint8_t board_error;
};
#pragma pack(pop)

class XbotGreenBridge {
public:
    explicit XbotGreenBridge(std::string interface_name)
        : interface_name_(std::move(interface_name)) {}

    ~XbotGreenBridge() { disconnect(); }

    void connect() {
        if (master_) {
            return;
        }
        master_ = std::make_shared<EtherCatMaster>(interface_name_, 1, false);
        if (!master_->open(false)) {
            master_.reset();
            throw std::runtime_error("failed to open EtherCAT master on " +
                                     interface_name_);
        }
        const auto slaves = master_->get_slave_list();
        for (std::size_t i = 0; i < slaves.size(); ++i) {
            const auto &slave = slaves[i].get();
            if (std::string(slave.name).find("HAND_GREEN") != std::string::npos) {
                slave_id_ = static_cast<std::uint8_t>(i + 1);
                slave_name_ = slave.name;
                if (slave.Ibytes < sizeof(GreenRxPdo) ||
                    slave.Obytes < sizeof(GreenJointCommand) * 3) {
                    disconnect();
                    throw std::runtime_error("unexpected XHand Green PDO size");
                }
                break;
            }
        }
        if (slave_id_ == 0) {
            disconnect();
            throw std::runtime_error("XHand Green EtherCAT slave not found");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        poll();
    }

    void disconnect() {
        if (master_) {
            master_->close();
            master_.reset();
        }
        slave_id_ = 0;
    }

    std::string slave_name() const { return slave_name_; }

    std::vector<std::tuple<float, float, float, int, int>>
    read_joint_diag() {
        poll();
        std::vector<std::tuple<float, float, float, int, int>> result;
        result.reserve(kJointCount);
        for (std::size_t i = 0; i < kJointCount; ++i) {
            if (!joint_seen_[i]) {
                throw std::runtime_error("missing joint " + std::to_string(i) +
                                         " in Green PDO");
            }
            const auto &s = joints_[i];
            result.emplace_back(
                ParamCodec::decode_position(s.position),
                ParamCodec::decode_speed(s.speed),
                ParamCodec::decode_torque(s.torque),
                static_cast<int>(s.temperature),
                static_cast<int>(s.error_code));
        }
        return result;
    }

    std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                           std::uint64_t, int>>
    read_sensor_diag() {
        poll();
        std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                               std::uint64_t, int>>
            result(kSensorSlots);
        for (std::size_t i = 0; i < kSensorSlots; ++i) {
            if (!sensor_seen_[i]) {
                continue;
            }
            const auto &s = sensors_[i];
            result[i] = {
                s.first_ydds_nf, s.second_ydds_nf,
                s.first_ydds_tf, s.second_ydds_tf,
                static_cast<int>(s.error_code)};
        }
        return result;
    }

    std::pair<int, int> board_diag() {
        poll();
        return {static_cast<int>(board_temperature_),
                static_cast<int>(board_error_)};
    }

private:
    void ensure_connected() const {
        if (!master_ || slave_id_ == 0) {
            throw std::runtime_error("XHand Green bridge is not connected");
        }
    }

    void poll() {
        ensure_connected();
        // Two complete sweeps cover the one-cycle request/response latency.
        for (std::size_t request = 0; request < 14; ++request) {
            const std::size_t start = (request * 3) % kJointCount;
            GreenJointCommand query[3]{};
            for (std::size_t i = 0; i < 3; ++i) {
                query[i].joint_id =
                    static_cast<std::uint16_t>((start + i) % kJointCount);
            }
            if (!master_->pdo_write(
                    slave_id_, reinterpret_cast<std::uint8_t *>(query),
                    sizeof(query))) {
                throw std::runtime_error("failed to send Green telemetry query");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));

            GreenRxPdo pdo{};
            if (master_->pdo_read(slave_id_,
                                  reinterpret_cast<std::uint8_t *>(&pdo),
                                  sizeof(pdo)) != sizeof(pdo)) {
                continue;
            }
            board_temperature_ = pdo.board_temperature;
            board_error_ = pdo.board_error;
            for (const auto &joint : pdo.joints) {
                if (joint.joint_id < kJointCount) {
                    joints_[joint.joint_id] = joint;
                    joint_seen_[joint.joint_id] = true;
                }
            }
            for (const auto &sensor : pdo.sensors) {
                if (sensor.sensor_id < kSensorSlots) {
                    sensors_[sensor.sensor_id] = sensor;
                    sensor_seen_[sensor.sensor_id] = true;
                }
            }
        }
    }

    std::string interface_name_;
    std::string slave_name_;
    std::shared_ptr<EtherCatMaster> master_;
    std::uint8_t slave_id_ = 0;
    std::array<GreenJointState, kJointCount> joints_{};
    std::array<bool, kJointCount> joint_seen_{};
    std::array<GreenSensor, kSensorSlots> sensors_{};
    std::array<bool, kSensorSlots> sensor_seen_{};
    std::uint8_t board_temperature_ = 0;
    std::uint8_t board_error_ = 0;
};

}  // namespace

PYBIND11_MODULE(xbot_green_bridge, m) {
    m.doc() = "Read-only XHand Green adapter for the installed /opt/xbot SDK";
    py::class_<XbotGreenBridge>(m, "XbotGreenBridge")
        .def(py::init<std::string>(), py::arg("interface_name") = "enp129s0")
        .def("connect", &XbotGreenBridge::connect)
        .def("disconnect", &XbotGreenBridge::disconnect)
        .def("slave_name", &XbotGreenBridge::slave_name)
        .def("read_joint_diag", &XbotGreenBridge::read_joint_diag)
        .def("read_sensor_diag", &XbotGreenBridge::read_sensor_diag)
        .def("board_diag", &XbotGreenBridge::board_diag);
}
