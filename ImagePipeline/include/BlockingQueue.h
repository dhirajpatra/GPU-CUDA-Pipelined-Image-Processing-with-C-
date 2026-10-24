// BlockingQueue.h - bounded multi-producer/multi-consumer queue used to hand
// frames from one pipeline stage to the next.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>

namespace ipp {

/// Thread-safe FIFO. close() lets consumers drain the remaining items and then
/// receive std::nullopt; abort() additionally discards everything (error path).
template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(std::size_t capacity = std::numeric_limits<std::size_t>::max())
        : capacity_(capacity) {}

    /// Blocks while full. Returns false if the queue was closed (item not enqueued).
    bool push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        notFull_.wait(lock, [&] { return closed_ || items_.size() < capacity_; });
        if (closed_) return false;
        items_.push_back(std::move(value));
        notEmpty_.notify_one();
        return true;
    }

    /// Blocks while empty. Returns std::nullopt once closed and drained.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        notEmpty_.wait(lock, [&] { return closed_ || !items_.empty(); });
        if (items_.empty()) return std::nullopt;
        T value = std::move(items_.front());
        items_.pop_front();
        notFull_.notify_one();
        return value;
    }

    /// No more pushes; consumers still receive what is already queued.
    void close() {
        { std::lock_guard<std::mutex> lock(mutex_); closed_ = true; }
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

    /// Close and discard queued items so every blocked thread wakes up immediately.
    void abort() {
        { std::lock_guard<std::mutex> lock(mutex_); closed_ = true; items_.clear(); }
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
    std::deque<T> items_;
    std::size_t capacity_;
    bool closed_ = false;
};

}  // namespace ipp
