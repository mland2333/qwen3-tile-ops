// sim_bridge: load a guest image plus a data blob and run it on Tile-NPU-Sim.
//
// The Mac checkout drove the simulator through a small harness that placed the
// blob at a fixed address and passed that address in x31; that tool was not
// part of any repository. This bridge restores the contract:
//
//   * guest binary at 0x80000000 (the simulator's entry),
//   * blob at --blob-base (default 0x81000000), also written to x31,
//   * run until kernel.end / ebreak, then dump the blob back to --out.
//
// Build: make bridge (needs a built Tile-NPU-Sim; see TILE_NPU_SIM).

#include "tile_npu/platform/bus.h"
#include "tile_npu/platform/ram.h"
#include "tile_npu/platform/serial.h"
#include "tile_npu/simulator.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t MEMORY_BASE = 0x80000000;
constexpr std::size_t DEFAULT_MEMORY_MIB = 1024;

bool parse_u64(std::string_view text, std::uint64_t &value) {
  try {
    std::size_t used = 0;
    value = std::stoull(std::string(text), &used, 0);
    return used == text.size();
  } catch (...) {
    return false;
  }
}

bool load_image(const std::string &path, std::uint64_t address, tile_npu::Ram &ram) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::cerr << "cannot open image: " << path << '\n';
    return false;
  }
  const auto size = static_cast<std::size_t>(input.tellg());
  std::vector<std::byte> bytes(size);
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size))) {
    std::cerr << "cannot read image: " << path << '\n';
    return false;
  }
  if (!ram.write(address, std::span<const std::byte>(bytes))) {
    std::cerr << "image does not fit in RAM at 0x" << std::hex << address << std::dec << ": "
              << path << " (" << size << " bytes)\n";
    return false;
  }
  return true;
}

bool dump_range(tile_npu::Ram &ram, std::uint64_t address, std::size_t size,
                const std::string &path) {
  std::vector<std::byte> bytes(size);
  if (!ram.read(address, std::span<std::byte>(bytes))) {
    std::cerr << "cannot dump memory at 0x" << std::hex << address << std::dec << '\n';
    return false;
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::cerr << "cannot open dump file: " << path << '\n';
    return false;
  }
  out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(size));
  return static_cast<bool>(out);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: sim_bridge <guest.bin> <blob.bin> <out.bin> "
                 "[--blob-base 0x81000000] [--memory MiB] [--steps N] [--dump FILE@OFF:SIZE]\n";
    return 2;
  }
  const std::string guest_path = argv[1];
  const std::string blob_path = argv[2];
  const std::string out_path = argv[3];
  std::uint64_t blob_base = 0x81000000;
  std::size_t memory_mib = DEFAULT_MEMORY_MIB;
  std::uint64_t step_limit = std::numeric_limits<std::uint64_t>::max();
  struct DumpSpec {
    std::string path;
    std::uint64_t offset;
    std::size_t size;
  };
  std::vector<DumpSpec> dumps;
  for (int i = 4; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--blob-base" && i + 1 < argc) {
      if (!parse_u64(argv[++i], blob_base)) {
        std::cerr << "invalid --blob-base\n";
        return 2;
      }
    } else if (arg == "--memory" && i + 1 < argc) {
      std::uint64_t mib = 0;
      if (!parse_u64(argv[++i], mib)) {
        std::cerr << "invalid --memory\n";
        return 2;
      }
      memory_mib = static_cast<std::size_t>(mib);
    } else if (arg == "--steps" && i + 1 < argc) {
      if (!parse_u64(argv[++i], step_limit)) {
        std::cerr << "invalid --steps\n";
        return 2;
      }
    } else if (arg == "--dump" && i + 1 < argc) {
      // --dump FILE@0xOFFSET:SIZE (the offset is relative to the blob base)
      const std::string spec = argv[++i];
      const auto at = spec.find('@');
      const auto colon = spec.find(':');
      if (at == std::string::npos || colon == std::string::npos || colon < at) {
        std::cerr << "invalid --dump (want FILE@OFF:SIZE)\n";
        return 2;
      }
      DumpSpec entry;
      entry.path = spec.substr(0, at);
      std::uint64_t size = 0;
      if (!parse_u64(spec.substr(at + 1, colon - at - 1), entry.offset) ||
          !parse_u64(spec.substr(colon + 1), size)) {
        std::cerr << "invalid --dump (want FILE@OFF:SIZE)\n";
        return 2;
      }
      entry.size = static_cast<std::size_t>(size);
      dumps.push_back(entry);
    } else {
      std::cerr << "unknown argument: " << arg << '\n';
      return 2;
    }
  }

  tile_npu::Ram ram(MEMORY_BASE, memory_mib << 20);
  if (!load_image(guest_path, MEMORY_BASE, ram)) return 1;
  std::ifstream blob_stream(blob_path, std::ios::binary | std::ios::ate);
  if (!blob_stream) {
    std::cerr << "cannot open blob: " << blob_path << '\n';
    return 1;
  }
  const auto blob_size = static_cast<std::size_t>(blob_stream.tellg());
  blob_stream.close();
  if (!load_image(blob_path, blob_base, ram)) return 1;

  tile_npu::SerialConsole serial(std::cout);
  tile_npu::SystemBus bus(ram, {{&serial, tile_npu::SerialConsole::BASE,
                                 tile_npu::SerialConsole::SIZE}});
  tile_npu::Simulator sim(bus);
  sim.reset(MEMORY_BASE);
  sim.state().scalar.write_x(31, blob_base);
  sim.state().scalar.write_x(30, 0); // stack top: set by the guest crt0

  std::uint64_t steps = 0;
  tile_npu::StepResult result{.status = tile_npu::StepStatus::retired, .pc = MEMORY_BASE};
  while (steps < step_limit) {
    result = sim.step();
    ++steps;
    if (result.status != tile_npu::StepStatus::retired) break;
  }

  const auto exit_code = static_cast<int>(sim.state().scalar.read_x(10));
  if (dumps.empty()) {
    DumpSpec all{out_path, 0, blob_size};
    dumps.push_back(all);
  }
  for (const auto &entry : dumps) {
    if (!dump_range(ram, blob_base + entry.offset, entry.size, entry.path)) return 1;
  }

  std::cerr << "stop: " << tile_npu::status_name(result.status) << " pc=0x" << std::hex
            << result.pc << std::dec << " steps=" << steps << " a0=" << exit_code << '\n';
  if (result.status == tile_npu::StepStatus::kernel_end ||
      result.status == tile_npu::StepStatus::breakpoint) {
    return exit_code;
  }
  return 1;
}
