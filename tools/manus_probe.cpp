#include "ManusSDK.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_running{true};
std::atomic<uint64_t> g_frames{0};
std::atomic<uint64_t> g_skeletons{0};
std::atomic<bool> g_glove_seen{false};
std::mutex g_print_mutex;

const char* return_code_name(SDKReturnCode code) {
    static const char* names[] = {
        "Success", "Error", "InvalidArgument", "ArgumentSizeMismatch",
        "UnsupportedStringSize", "SdkNotAvailable", "HostFinderNotAvailable",
        "DataNotAvailable", "MemoryError", "InternalError",
        "FunctionCalledAtWrongTime", "NotConnected", "ConnectionTimeout",
        "InvalidID", "NullPointer", "InvalidSequence", "NoCoordinateSystemSet",
        "SdkIsTerminating", "StubNullPointer", "SkeletonNotLoaded",
        "FunctionNotAvailable",
    };
    const auto index = static_cast<int>(code);
    return index >= 0 && index < static_cast<int>(sizeof(names) / sizeof(names[0]))
               ? names[index]
               : "Unknown";
}

const char* side_name(Side side) {
    switch (side) {
        case Side_Left: return "left";
        case Side_Right: return "right";
        case Side_Center: return "center";
        default: return "invalid";
    }
}

const char* family_name(DeviceFamilyType family) {
    switch (family) {
        case DeviceFamilyType_Prime1: return "Prime1";
        case DeviceFamilyType_Prime2: return "Prime2";
        case DeviceFamilyType_PrimeX: return "PrimeX";
        case DeviceFamilyType_Metaglove: return "Metaglove";
        case DeviceFamilyType_Prime3: return "Prime3";
        case DeviceFamilyType_Virtual: return "Virtual";
        case DeviceFamilyType_MetaglovePro: return "MetaglovePro";
        case DeviceFamilyType_MetagloveProPrecision: return "MetagloveProPrecision";
        case DeviceFamilyType_MetagloveProHaptics: return "MetagloveProHaptics";
        case DeviceFamilyType_MetagloveProPrecisionHaptics:
            return "MetagloveProPrecisionHaptics";
        default: return "Unknown";
    }
}

const char* chain_name(ChainType chain) {
    switch (chain) {
        case ChainType_Hand: return "hand";
        case ChainType_FingerThumb: return "thumb";
        case ChainType_FingerIndex: return "index";
        case ChainType_FingerMiddle: return "middle";
        case ChainType_FingerRing: return "ring";
        case ChainType_FingerPinky: return "pinky";
        default: return "other";
    }
}

const char* joint_name(FingerJointType joint) {
    switch (joint) {
        case FingerJointType_Metacarpal: return "metacarpal";
        case FingerJointType_Proximal: return "proximal";
        case FingerJointType_Intermediate: return "intermediate";
        case FingerJointType_Distal: return "distal";
        case FingerJointType_Tip: return "tip";
        default: return "n/a";
    }
}

void signal_handler(int) { g_running.store(false); }

void log_callback(LogSeverity severity, const char* log, uint32_t length) {
    const char* level = severity == LogSeverity_Error   ? "ERROR"
                        : severity == LogSeverity_Warn ? "WARN"
                        : severity == LogSeverity_Info ? "INFO"
                                                       : "DEBUG";
    std::lock_guard<std::mutex> lock(g_print_mutex);
    std::cout << "[sdk " << level << "] " << std::string(log, length) << '\n';
}

void landscape_callback(const Landscape* landscape) {
    static uint32_t last_glove_count = UINT32_MAX;
    static uint32_t last_dongle_count = UINT32_MAX;
    if (!landscape) return;
    const auto& devices = landscape->gloveDevices;
    if (devices.gloveCount == last_glove_count &&
        devices.dongleCount == last_dongle_count) {
        return;
    }
    last_glove_count = devices.gloveCount;
    last_dongle_count = devices.dongleCount;
    g_glove_seen.store(devices.gloveCount > 0);
    std::lock_guard<std::mutex> lock(g_print_mutex);
    std::cout << "[landscape] dongles=" << devices.dongleCount
              << " gloves=" << devices.gloveCount << '\n';
    for (uint32_t i = 0; i < devices.dongleCount; ++i) {
        const auto& d = devices.dongles[i];
        std::cout << "  dongle id=" << d.id << " family=" << family_name(d.familyType)
                  << " license='" << d.licenseType << "' max_pairs="
                  << d.licenseMaxNumberOfGlovePairs << " left=" << d.leftGloveID
                  << " right=" << d.rightGloveID << '\n';
    }
    for (uint32_t i = 0; i < devices.gloveCount; ++i) {
        const auto& glove = devices.gloves[i];
        std::cout << "  glove id=" << glove.id << " side=" << side_name(glove.side)
                  << " family=" << family_name(glove.familyType)
                  << " paired_state=" << static_cast<int>(glove.pairedState)
                  << " dongle=" << glove.dongleID
                  << " battery=" << glove.batteryPercentage << "%"
                  << " signal=" << glove.transmissionStrength << '\n';
    }
}

void raw_skeleton_callback(const SkeletonStreamInfo* stream) {
    if (!stream) return;
    const uint64_t frame = g_frames.fetch_add(1) + 1;
    g_skeletons.fetch_add(stream->skeletonsCount);
    if (stream->skeletonsCount == 0 || (frame != 1 && frame % 120 != 0)) return;

    RawSkeletonInfo info{};
    SDKReturnCode code = CoreSdk_GetRawSkeletonInfo(0, &info);
    if (code != SDKReturnCode_Success) return;
    std::vector<SkeletonNode> nodes(info.nodesCount);
    code = CoreSdk_GetRawSkeletonData(0, nodes.data(), info.nodesCount);
    if (code != SDKReturnCode_Success || nodes.empty()) return;

    std::lock_guard<std::mutex> lock(g_print_mutex);
    const auto& wrist = nodes.front().transform;
    std::cout << std::fixed << std::setprecision(5)
              << "[frame " << frame << "] glove=" << info.gloveId
              << " skeletons=" << stream->skeletonsCount
              << " nodes=" << info.nodesCount << " first_pos=("
              << wrist.position.x << ',' << wrist.position.y << ',' << wrist.position.z
              << ") first_quat=(" << wrist.rotation.w << ',' << wrist.rotation.x << ','
              << wrist.rotation.y << ',' << wrist.rotation.z << ")\n";

    if (frame == 1) {
        uint32_t node_count = 0;
        if (CoreSdk_GetRawSkeletonNodeCount(info.gloveId, node_count) ==
                SDKReturnCode_Success &&
            node_count > 0) {
            std::vector<NodeInfo> hierarchy(node_count);
            if (CoreSdk_GetRawSkeletonNodeInfoArray(info.gloveId, hierarchy.data(),
                                                    node_count) ==
                SDKReturnCode_Success) {
                std::cout << "[hierarchy] nodes=" << node_count << '\n';
                for (const auto& node : hierarchy) {
                    std::cout << "  id=" << node.nodeId << " parent=" << node.parentId
                              << " side=" << side_name(node.side)
                              << " chain=" << chain_name(node.chainType)
                              << " joint=" << joint_name(node.fingerJointType) << '\n';
                }
            }
        }
    }
}

bool require_success(const char* operation, SDKReturnCode code) {
    std::cout << '[' << operation << "] code=" << static_cast<int>(code) << " ("
              << return_code_name(code) << ")\n";
    return code == SDKReturnCode_Success;
}

}  // namespace

int main(int argc, char** argv) {
    int duration_seconds = 15;
    if (argc > 1) duration_seconds = std::max(1, std::stoi(argv[1]));
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    SDKReturnCode code = CoreSdk_InitializeIntegrated();
    if (!require_success("initialize_integrated", code)) return 2;

    CoreSdk_RegisterCallbackForOnLog(log_callback);
    const std::string settings_path =
        std::filesystem::absolute("build/manus_probe/runtime/settings").string() + "/";
    const std::string log_path =
        std::filesystem::absolute("build/manus_probe/runtime/logs").string() + "/";
    require_success("settings_location",
                    CoreSdk_SetSettingsLocation(settings_path.c_str()));
    require_success("log_location", CoreSdk_SetLogLocation(log_path.c_str()));
    CoreSdk_RegisterCallbackForLandscapeStream(landscape_callback);
    CoreSdk_RegisterCallbackForRawSkeletonStream(raw_skeleton_callback);

    CoordinateSystemVUH coordinates{};
    CoordinateSystemVUH_Init(&coordinates);
    coordinates.handedness = Side_Right;
    coordinates.up = AxisPolarity_PositiveZ;
    coordinates.view = AxisView_XFromViewer;
    coordinates.unitScale = 1.0f;
    code = CoreSdk_InitializeCoordinateSystemWithVUH(coordinates, false);
    if (!require_success("coordinate_system", code)) {
        CoreSdk_ShutDown();
        return 3;
    }

    // Integrated mode discovers USB devices asynchronously.  This older SDK
    // build crashes if SetRawSkeletonHandMotion is called before the first
    // glove has appeared in the landscape, so wait for discovery first.
    const auto discovery_deadline = std::chrono::steady_clock::now() +
                                    std::chrono::seconds(5);
    while (!g_glove_seen.load() &&
           std::chrono::steady_clock::now() < discovery_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (g_glove_seen.load()) {
        code = CoreSdk_SetRawSkeletonHandMotion(HandMotion_Auto);
        require_success("raw_skeleton_hand_motion_auto", code);
    } else {
        std::cout << "[probe] no glove appeared in the SDK landscape within 5s; "
                     "skipping hand-motion configuration\n";
    }

    std::cout << "[probe] waiting " << duration_seconds
              << " seconds; move and flex the connected hand (Ctrl-C to stop)\n";
    const auto start = std::chrono::steady_clock::now();
    while (g_running.load() &&
           std::chrono::steady_clock::now() - start <
               std::chrono::seconds(duration_seconds)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    const uint64_t frames = g_frames.load();
    std::cout << std::fixed << std::setprecision(2)
              << "[summary] elapsed=" << elapsed << "s frames=" << frames
              << " rate=" << (elapsed > 0.0 ? frames / elapsed : 0.0)
              << "Hz skeleton_samples=" << g_skeletons.load() << '\n';
    code = CoreSdk_ShutDown();
    require_success("shutdown", code);
    return frames > 0 ? 0 : 4;
}
