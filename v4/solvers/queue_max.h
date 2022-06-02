#include <deque>

template<typename T>
class QueueMax {
public:
    struct S {
        T value;
        S* next = nullptr;
        S* prev = nullptr;
        S(const T& value) : value(value) {}
    };
    std::deque<S> queue;
    S* max = nullptr;
public:
    void Push(const T& value) {
        S cur(value);
        if (!queue.empty()) {
            cur.prev = &(queue.back());
        }
        queue.push_back(cur);
        while (queue.back().prev && queue.back().prev->value < cur.value) {
            queue.back().prev = queue.back().prev->prev;
        }
        if (queue.back().prev) {
            queue.back().prev->next = &queue.back();
        } else {
            max = &queue.back();
        }
    }

    void Pop() {
        if (&queue.front() == max) {
            max = queue.front().next;
            if (max) {
                max->prev = nullptr;
            }
        }
        return queue.pop_front();
    }

    const T& Back() const {
        return queue.back().value;
    }

    const T& Front() const {
        return queue.front().value;
    }

    const T& Max() const {
        return max->value;
    }

    size_t Size() const {
        return queue.size();
    }
};
#include <set>

template<typename T>
class QueueMin {
public:
    std::deque<T> queue;
    std::multiset<T> set;

public:
    void Push(const T& value) {
        queue.push_back(value);
        set.insert(value);
    }

    void Pop() {
        set.erase(set.find(queue.front()));
        queue.pop_front();
    }

    const T& Back() const {
        return queue.back();
    }

    const T& Front() const {
        return queue.front();
    }

    const T& Min() const {
        return *set.begin();
    }

    size_t Size() const {
        return queue.size();
    }
};
