#pragma once

#ifdef __ANDROID__

#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "Logging.hpp"

enum EventID {
  // Cache Events : Some may not be suppoted on specific platform
  L1D_READ_ACCESS,
  L1D_READ_MISS,
  L1D_WRITE_ACCESS,
  L1D_WRITE_MISS,
  L1D_PREFETCH_ACCESS,
  L1D_PREFETCH_MISS,

  L1I_READ_ACCESS,
  L1I_READ_MISS,
  L1I_PREFETCH_ACCESS,
  L1I_PREFETCH_MISS,

  LLC_READ_ACCESS,
  LLC_READ_MISS,
  LLC_WRITE_ACCESS,
  LLC_WRITE_MISS,
  LLC_PREFETCH_ACCESS,
  LLC_PREFETCH_MISS,

  DTLB_READ_ACCESS,
  DTLB_READ_MISS,
  DTLB_WRITE_ACCESS,
  DTLB_WRITE_MISS,
  DTLB_PREFETCH_ACCESS,
  DTLB_PREFETCH_MISS,

  ITLB_READ_ACCESS,
  ITLB_READ_MISS,
  ITLB_PREFETCH_ACCESS,
  ITLB_PREFETCH_MISS,

  BPU_READ_ACCESS,
  BPU_READ_MISS,
  BPU_PREFETCH_ACCESS,
  BPU_PREFETCH_MISS,

  NODE_READ_ACCESS,
  NODE_READ_MISS,
  NODE_WRITE_ACCESS,
  NODE_WRITE_MISS,
  NODE_PREFETCH_ACCESS,
  NODE_PREFETCH_MISS,

  // Some useful events
  CPU_CYCLES,
  CPU_INSTRUCTIONS,
  TASK_CLOCK,
  PAGE_FAULTS,
  STALLED_CYCLES_FRONTEND,
  STALLED_CYCLES_BACKEND,
};

// No use of sampling,group
// Mesures calling process/thread on any cpu
class SimplePerf {
 public:
  SimplePerf() {
      // Enable profiling on device
      std::system("setprop security.perf_harden 0");
  }

 public:
  void Start(bool useDefaultEvents = true,
             std::initializer_list<EventID> eventIds = std::initializer_list<EventID>());
  void Stop();
  void Report();
  void Reset();
  int64_t GetResult(EventID id, int scale = 100);

 private:
  struct EventInfo {
    struct Result {
      uint64_t value = -1;
      uint64_t time_enabled = -1;
      uint64_t time_running = -1;
    };
    EventID id;
    perf_event_attr pe;
    int fd;
    Result result;

    double GetResult(int scale = 100) const;
    std::string Format() const;
  };
  struct EncodedEvent {
    std::string name;
    uint32_t type;
    uint64_t id;
  };

 private:
  std::vector<std::pair<EventInfo, bool>> cacheEvents;
  static std::map<EventID, EncodedEvent> EventMap;
  std::chrono::time_point<std::chrono::high_resolution_clock> timeBegin, timeEnd;

 private:
  void DefaultEvents();
  void AddEvent(EventID id);
};

double SimplePerf::EventInfo::GetResult(int scale) const {
    return result.value * (result.time_running / double(result.time_enabled)) / double(scale);
}
std::string SimplePerf::EventInfo::Format() const {
    std::string str = EventMap[id].name;
    str.resize(20, ' ');
    str += std::to_string(GetResult());
    return str;
}

void SimplePerf::Start(bool useDefaultEvents, std::initializer_list<EventID> eventIds) {
    cacheEvents.clear();
    if (useDefaultEvents) {
        DefaultEvents();
    } else {
        for (const auto &eventId: eventIds) {
            AddEvent(eventId);
        }
    }
    for (auto &event: cacheEvents) {
        auto &eventInfo = event.first;
        int fd = syscall(__NR_perf_event_open, &eventInfo.pe, 0, -1, -1, 0);
        if (fd < 0) {
            // Not supported event
            event.second = false;
        } else {
            eventInfo.fd = fd;
            event.second = true;
            ioctl(eventInfo.fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(eventInfo.fd, PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    timeBegin = std::chrono::high_resolution_clock::now();
}

void SimplePerf::Stop() {
    timeEnd = std::chrono::high_resolution_clock::now();
    for (auto &event: cacheEvents) {
        if (event.second) {
            auto &eventInfo = event.first;
            ioctl(eventInfo.fd, PERF_EVENT_IOC_DISABLE, 0);
            if (read(eventInfo.fd, &eventInfo.result, sizeof(uint64_t) * 3)
                != sizeof(uint64_t) * 3) {
                std::cout << "Failed to read the result of event :" << EventMap[eventInfo.id].name
                          << std::endl;
                std::cout << "Error description : " << strerror(errno) << std::endl;
            }
            close(event.first.fd);
        }
    }
}

void SimplePerf::Report() {
    using namespace PNN;
    LOGI("--------------------------------");
    LOGI("Result: \n");
    LOGI("TimeConsume: %d ms",
         std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeBegin).count());
//    std::cout << "Result:" << std::endl;
//    std::cout << "Time Consume: " << std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeBegin).count()
//              << " ms " << std::endl;

    std::vector<EventID> unsported;
    for (const auto &event: cacheEvents) {
        if (event.second) {
            LOGI("%s", event.first.Format().c_str());
            std::cout << event.first.Format() << std::endl;
        } else {
            unsported.push_back(event.first.id);
        }
    }
    if (!unsported.empty()) {
        std::cout << "Unsupported Events:";
        for (const auto &e: unsported) {
            std::cout << EventMap[e].name << " ";
        }
        std::cout << std::endl;
    }
}

void SimplePerf::AddEvent(EventID id) {
    EventInfo eventInfo;
    eventInfo.id = id;
    auto &pe = eventInfo.pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
    // Fixed Attr
    pe.type = EventMap[id].type;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = EventMap[id].id;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    cacheEvents.push_back({eventInfo, false});
}

void SimplePerf::DefaultEvents() {
    AddEvent(EventID::L1D_READ_ACCESS);
    AddEvent(EventID::L1D_READ_MISS);
    AddEvent(EventID::BPU_READ_ACCESS);
    AddEvent(EventID::STALLED_CYCLES_FRONTEND);
}

int64_t SimplePerf::GetResult(EventID id, int scale) {
    auto iter = std::find_if(std::begin(cacheEvents), std::end(cacheEvents),
                             [&](const std::pair<EventInfo, bool> &p) { return p.first.id == id; });
    if (iter == std::end(cacheEvents)) {
        return -1;
    } else {
        return iter->first.GetResult(scale);
    }
}

std::map<EventID, SimplePerf::EncodedEvent> SimplePerf::EventMap = {
    {L1D_READ_ACCESS,
     {"L1D_READ_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_L1D | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {L1D_READ_MISS,
     {"L1D_READ_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_L1D | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
    {L1D_WRITE_ACCESS,
     {"L1D_WRITE_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_L1D | PERF_COUNT_HW_CACHE_OP_WRITE << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {L1D_WRITE_MISS,
     {"L1D_WRITE_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_L1D | PERF_COUNT_HW_CACHE_OP_WRITE << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
    {L1D_PREFETCH_ACCESS,
     {"L1D_PREFETCH_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_L1D | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {L1D_PREFETCH_MISS,
     {"L1D_PREFETCH_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_L1D | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
#pragma mark

    {L1I_READ_ACCESS,
     {"L1I_READ_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_L1I | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {L1I_READ_MISS,
     {"L1I_READ_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_L1I | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},

    {L1I_PREFETCH_ACCESS,
     {"L1I_PREFETCH_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_L1I | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {L1I_PREFETCH_MISS,
     {"L1I_PREFETCH_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_L1I | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
#pragma mark

    {LLC_READ_ACCESS,
     {"LLC_READ_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_LL | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {LLC_READ_MISS,
     {"LLC_READ_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_LL | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
    {LLC_WRITE_ACCESS,
     {"LLC_WRITE_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_LL | PERF_COUNT_HW_CACHE_OP_WRITE << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {LLC_WRITE_MISS,
     {"LLC_WRITE_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_LL | PERF_COUNT_HW_CACHE_OP_WRITE << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
    {LLC_PREFETCH_ACCESS,
     {"LLC_PREFETCH_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_LL | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {LLC_PREFETCH_MISS,
     {"LLC_PREFETCH_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_LL | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},

#pragma mark

    {DTLB_READ_ACCESS,
     {"DTLB_READ_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_DTLB | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {DTLB_READ_MISS,
     {"DTLB_READ_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_DTLB | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
    {DTLB_WRITE_ACCESS,
     {"DTLB_WRITE_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_DTLB | PERF_COUNT_HW_CACHE_OP_WRITE << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {DTLB_WRITE_MISS,
     {"DTLB_WRITE_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_DTLB | PERF_COUNT_HW_CACHE_OP_WRITE << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
    {DTLB_PREFETCH_ACCESS,
     {"DTLB_PREFETCH_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_DTLB | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {DTLB_PREFETCH_MISS,
     {"DTLB_PREFETCH_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_DTLB | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
#pragma mark

    {ITLB_READ_ACCESS,
     {"ITLB_READ_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_ITLB | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {ITLB_READ_MISS,
     {"ITLB_READ_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_ITLB | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},

    {ITLB_PREFETCH_ACCESS,
     {"ITLB_PREFETCH_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_ITLB | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {ITLB_PREFETCH_MISS,
     {"ITLB_PREFETCH_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_ITLB | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
#pragma mark
    {BPU_READ_ACCESS,
     {"BPU_READ_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_BPU | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {BPU_READ_MISS,
     {"BPU_READ_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_BPU | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},

    {BPU_PREFETCH_ACCESS,
     {"BPU_PREFETCH_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_BPU | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {BPU_PREFETCH_MISS,
     {"BPU_PREFETCH_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_BPU | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
#pragma mark
    {NODE_READ_ACCESS,
     {"NODE_READ_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_NODE | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {NODE_READ_MISS,
     {"NODE_READ_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_NODE | PERF_COUNT_HW_CACHE_OP_READ << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
    {NODE_WRITE_ACCESS,
     {"NODE_WRITE_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_NODE | PERF_COUNT_HW_CACHE_OP_WRITE << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {NODE_WRITE_MISS,
     {"NODE_WRITE_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_NODE | PERF_COUNT_HW_CACHE_RESULT_MISS << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},
    {NODE_PREFETCH_ACCESS,
     {"NODE_PREFETCH_ACCESS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_NODE | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16}},
    {NODE_PREFETCH_MISS,
     {"NODE_PREFETCH_MISS", PERF_TYPE_HW_CACHE,
      PERF_COUNT_HW_CACHE_NODE | PERF_COUNT_HW_CACHE_OP_PREFETCH << 8
          | PERF_COUNT_HW_CACHE_RESULT_MISS << 16}},

#pragma mark
    {CPU_CYCLES, {"CPU_CYCLES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES}},
    {CPU_INSTRUCTIONS, {"CPU_INSTRUCTIONS", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS}},
    {TASK_CLOCK, {"TASK_CLOCK", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK}},
    {PAGE_FAULTS, {"PAGE_FAULTS", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS}},
    {STALLED_CYCLES_FRONTEND,
     {"STALLED_CYCLES_FRONTEND", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND}},
    {STALLED_CYCLES_BACKEND,
     {"STALLED_CYCLES_BACKEND", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND}}};

#endif
