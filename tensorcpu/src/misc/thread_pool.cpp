//
// Created by tejag on 2024-05-01.
//

#include <cstdint>
#include <cstring>
#include <functional>
#include <pthread.h>
#include <queue>
#include <thread>
#include <unistd.h>
#include <vector>

#include "thread_pool.hpp"
#include "debug.hpp"

ThreadPool pool;

int setHighestThreadPriority(pthread_t thId) {
  int err = nice(-19);
  if (err == -1) {
    if (kDebug && kDebugLevel >= kDebugLevelVerbose) {
      std::cerr << __FILE__ << ":" << __LINE__ << " => Failed to set nice value" << std::endl;
    }
  }
  pthread_attr_t thAttr;
  err = pthread_attr_init(&thAttr);
  if (err != 0) return err;
  int policicies[3] = {SCHED_FIFO, SCHED_RR, SCHED_OTHER};
  int i = 0;
  for (; i < 3; i++) {
    int policy = policicies[i];
    err = pthread_attr_setschedpolicy(&thAttr, policy);
    if (err != 0) continue;
    err = pthread_attr_getschedpolicy(&thAttr, &policy);
    if (err != 0) return err;
    int maxPrio = sched_get_priority_max(policy);

    sched_param sp;
    memset(&sp, 0, sizeof(struct sched_param));
    sp.sched_priority = maxPrio;

    err = pthread_setschedparam(thId, policy, &sp);
    if (err != 0) continue;

    if(kDebug && kDebugLevel >= kDebugLevelVerbose) {
      std::cerr << "Successfully set scheduling params; policy: " << policy
                << " priority: " << maxPrio << std::endl;
    }

    break;
  }
  err = pthread_attr_destroy(&thAttr);
  if (err != 0) return err;
  if (i == 3) return -1;
  return 0;
}

#if not defined(__APPLE__)
#else
#include <sys/sysctl.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#include <pthread.h>

#define SYSCTL_CORE_COUNT "machdep.cpu.core_count"

typedef struct cpu_set {
  uint32_t count;
} cpu_set_t;

static inline void CPU_ZERO(cpu_set_t *cs) { cs->count = 0; }

static inline void CPU_SET(int num, cpu_set_t *cs) { cs->count |= (1 << num); }

static inline int CPU_ISSET(int num, cpu_set_t *cs) {
  return (cs->count & (1 << num));
}

int sched_getaffinity(pid_t pid, size_t cpu_size, cpu_set_t *cpu_set) {
  int32_t core_count = 0;
  size_t len = sizeof(core_count);
  int ret = sysctlbyname(SYSCTL_CORE_COUNT, &core_count, &len, nullptr, 0);
  if (ret) {
    printf("error while get core count %d\n", ret);
    return -1;
  }
  cpu_set->count = 0;
  for (int i = 0; i < core_count; i++) {
    cpu_set->count |= (1 << i);
  }

  return 0;
}

int pthread_setaffinity_np(
    pthread_t thread, size_t cpu_size, cpu_set_t *cpu_set
) {
  thread_port_t mach_thread;
  int core = 0;

  for (core = 0; core < 8 * cpu_size; core++) {
    if (CPU_ISSET(core, cpu_set)) break;
  }
  thread_affinity_policy_data_t policy = {core};
  mach_thread = pthread_mach_thread_np(thread);
  thread_policy_set(
      mach_thread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, 1
  );
  return 0;
}
#endif

int pinThreadToCore(pthread_t thread, int coreId) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreId, &cpuset);
  return pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}