// Read-only XHand Green EtherCAT PDO diagnostic.
//
// This intentionally does not construct EthercatGreenHand because that class
// starts a command-sending thread in its constructor.  Green's cyclic protocol
// requires the master to put three requested joint IDs in each output PDO.  We
// therefore send ID-only query frames with mode/kp/kd/position/speed/torque all
// zero.  Mode 0 is neither position mode (3) nor MIT mode (4), so these frames
// select telemetry pages without issuing a motion command.

#include <soem_cpp/ec_master.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "ethercat_drive/hand/param_codec.h"

namespace {

constexpr std::size_t kJointCount = 21;

#pragma pack(push, 1)
struct GreenJointState {
    std::uint16_t joint_id;
    std::int16_t position;
    std::int16_t speed;
    std::int16_t torque;
    std::uint16_t error_code;
    std::int8_t temperature;
    std::uint8_t reserved;
};

struct GreenJointQuery {
    std::uint16_t joint_id;
    std::uint8_t mode;
    std::uint8_t reserved;
    std::uint16_t kp;
    std::uint16_t kd;
    std::int16_t position;
    std::int16_t speed;
    std::int16_t torque;
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
    std::uint8_t communication_board_temperature;
    std::uint8_t communication_board_error;
};
#pragma pack(pop)

constexpr std::array<const char *, kJointCount> kJointNames = {
    "hand_thumb_bend_joint",
    "hand_thumb_rota_joint1",
    "hand_thumb_rota_joint2",
    "hand_thumb_rota_joint3",
    "hand_index_bend_joint",
    "hand_index_rota_joint1",
    "hand_index_rota_joint2",
    "hand_index_rota_joint3",
    "hand_mid_bend_joint",
    "hand_mid_rota_joint1",
    "hand_mid_rota_joint2",
    "hand_mid_rota_joint3",
    "hand_ring_bend_joint",
    "hand_ring_rota_joint1",
    "hand_ring_rota_joint2",
    "hand_ring_rota_joint3",
    "hand_pinky_bend_joint",
    "hand_pinky_rota_joint1",
    "hand_pinky_rota_joint2",
    "hand_pinky_rota_joint3",
    "hand_ltspalm_joint",
};

struct JointSample {
    bool seen = false;
    std::uint64_t samples = 0;
    GreenJointState state{};
};

}  // namespace

int main(int argc, char **argv) {
    const std::string interface_name = argc > 1 ? argv[1] : "enp129s0";
    const int duration_ms = argc > 2 ? std::max(100, std::atoi(argv[2])) : 3000;

    auto master = std::make_shared<EtherCatMaster>(interface_name, 1, false);
    if (!master->open(false)) {
        std::cerr << "[FAIL] Could not open EtherCAT master on "
                  << interface_name << '\n';
        return 2;
    }

    const auto slaves = master->get_slave_list();
    if (slaves.empty()) {
        std::cerr << "[FAIL] No EtherCAT slaves found\n";
        return 3;
    }

    std::uint8_t green_slave_id = 0;
    for (std::size_t i = 0; i < slaves.size(); ++i) {
        const auto &slave = slaves[i].get();
        std::cout << "[SLAVE] id=" << (i + 1) << " name=" << slave.name
                  << " state=0x" << std::hex << static_cast<int>(slave.state)
                  << std::dec << " ibytes=" << slave.Ibytes
                  << " obytes=" << slave.Obytes << '\n';
        if (std::string(slave.name).find("HAND_GREEN") != std::string::npos) {
            green_slave_id = static_cast<std::uint8_t>(i + 1);
        }
    }

    if (green_slave_id == 0) {
        std::cerr << "[FAIL] No XHand Green slave found\n";
        return 4;
    }

    const auto &green_slave = slaves[green_slave_id - 1].get();
    if (green_slave.Ibytes < sizeof(GreenRxPdo)) {
        std::cerr << "[FAIL] Green input PDO is " << green_slave.Ibytes
                  << " bytes, expected at least " << sizeof(GreenRxPdo) << '\n';
        return 5;
    }

    std::array<JointSample, kJointCount> samples{};
    std::uint8_t board_temperature = 0;
    std::uint8_t board_error = 0;
    std::uint64_t frames = 0;
    std::uint64_t invalid_joint_ids = 0;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(duration_ms);

    // EtherCatMaster creates its per-slave double buffers in the cyclic thread
    // immediately after open(); allow that first cycle to complete.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    while (std::chrono::steady_clock::now() < deadline) {
        static std::size_t query_start = 0;
        GreenJointQuery query[3]{};
        for (std::size_t i = 0; i < 3; ++i) {
            query[i].joint_id =
                static_cast<std::uint16_t>((query_start + i) % kJointCount);
        }
        if (!master->pdo_write(green_slave_id,
                               reinterpret_cast<std::uint8_t *>(&query),
                               sizeof(query))) {
            std::cerr << "[FAIL] Could not send ID-only telemetry query\n";
            return 6;
        }
        query_start = (query_start + 3) % kJointCount;
        std::this_thread::sleep_for(std::chrono::milliseconds(3));

        GreenRxPdo pdo{};
        if (master->pdo_read(green_slave_id,
                             reinterpret_cast<std::uint8_t *>(&pdo),
                             sizeof(pdo)) == sizeof(pdo)) {
            ++frames;
            board_temperature = pdo.communication_board_temperature;
            board_error = pdo.communication_board_error;
            for (const auto &joint : pdo.joints) {
                if (joint.joint_id >= kJointCount) {
                    ++invalid_joint_ids;
                    continue;
                }
                auto &sample = samples[joint.joint_id];
                sample.seen = true;
                ++sample.samples;
                sample.state = joint;
            }
        }
    }

    std::size_t seen_count = 0;
    std::size_t joint_error_count = 0;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[BOARD] temperature=" << static_cast<int>(board_temperature)
              << "C error=" << static_cast<int>(board_error)
              << " frames=" << frames
              << " invalid_joint_ids=" << invalid_joint_ids << '\n';
    std::cout << "id  joint_name                         position(rad)"
                 "  speed(rad/s)  torque(Nm)  temp(C)  error  samples\n";

    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto &sample = samples[i];
        if (!sample.seen) {
            std::cout << std::setw(2) << i << "  " << std::left
                      << std::setw(34) << kJointNames[i] << std::right
                      << "  MISSING\n";
            continue;
        }
        ++seen_count;
        if (sample.state.error_code != 0) {
            ++joint_error_count;
        }
        std::cout << std::setw(2) << i << "  " << std::left
                  << std::setw(34) << kJointNames[i] << std::right
                  << "  " << std::setw(12)
                  << ParamCodec::decode_position(sample.state.position)
                  << "  " << std::setw(12)
                  << ParamCodec::decode_speed(sample.state.speed)
                  << "  " << std::setw(10)
                  << ParamCodec::decode_torque(sample.state.torque)
                  << "  " << std::setw(7)
                  << static_cast<int>(sample.state.temperature)
                  << "  " << std::setw(5) << sample.state.error_code
                  << "  " << sample.samples << '\n';
    }

    const bool pass = frames > 0 && seen_count == kJointCount &&
                      invalid_joint_ids == 0 && board_error == 0 &&
                      joint_error_count == 0;
    std::cout << (pass ? "[PASS] " : "[FAIL] ")
              << "seen=" << seen_count << '/' << kJointCount
              << " joint_errors=" << joint_error_count
              << " board_error=" << static_cast<int>(board_error) << '\n';
    return pass ? 0 : 6;
}
