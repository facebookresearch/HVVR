#pragma once

/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace hvvr {

class ThreadPool {
public:

    typedef void(*InitCallback)(size_t threadIndex);

    ThreadPool(size_t numThreads, InitCallback initCallback = nullptr) : _run(true) {
        for (size_t n = 0; n < numThreads; n++) {
            _threads.emplace_back(&ThreadPool::threadEntry, this, initCallback, n);
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _run = false;
        }
        _condition.notify_all();
        for (std::thread& t : _threads) {
            t.join();
        }
    }

    template <typename Func, typename... Args>
    auto addTask(Func&& f, Args&&... args) {
        using return_type = typename std::result_of<Func(Args...)>::type;

        auto packagedFunc = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<Func>(f), std::forward<Args>(args)...));
        auto future = packagedFunc->get_future();
        auto task = [packagedFunc = std::move(packagedFunc)]() {(*packagedFunc)();};

        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_run) {
                _tasks.emplace(std::move(task));
            } else {
                assert(false); // tried to create a task after shutting down the thread pool
            }
        }
        _condition.notify_one();

        return future;
    }

protected:

    std::vector<std::thread> _threads;
    std::queue<std::function<void()>> _tasks;
    std::mutex _mutex;
    std::condition_variable _condition;
    std::atomic<bool> _run;

    void threadEntry(InitCallback initCallback, size_t threadIndex) {
        if (initCallback)
            initCallback(threadIndex);

        while (_run) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _condition.wait(lock,
                    [this]() {
                    return !_run || !_tasks.empty();
                });
                // even if we've been signalled to quit, finish any remaining tasks
                if (!_run && _tasks.empty()) {
                    break;
                }
                task = std::move(_tasks.front());
                _tasks.pop();
            }
            task();
        }
    }
};

} // namespace hvvr
