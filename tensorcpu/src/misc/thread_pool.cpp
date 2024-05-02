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

ThreadPool pool;

int setHighestThreadPriority(pthread_t thId) {
  int err = nice(-20);
  if (err == -1)
    return err;
  pthread_attr_t thAttr;
  err = pthread_attr_init(&thAttr);
  if (err != 0)
    return err;
  int policy = SCHED_OTHER;
  err = pthread_attr_setschedpolicy(&thAttr, policy);
  if (err != 0)
    return err;
  // err = pthread_attr_getschedpolicy(&thAttr, &policy);
  // if (err != 0) return err;
  int maxPrio = sched_get_priority_max(policy);
  std::cout << "policy: " << policy << " maxPrio: " << maxPrio << std::endl;

  sched_param sp;
  memset(&sp, 0, sizeof(struct sched_param));
  sp.sched_priority = maxPrio;

  err = pthread_setschedparam(thId, policy, &sp);
  if (err != 0)
    return err;
  err = pthread_attr_destroy(&thAttr);
  return err;
  return 0;
}