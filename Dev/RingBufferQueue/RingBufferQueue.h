#include "opencv2/core.hpp"
#include <cstddef>
#include <mutex>

#define BUFFER_SIZE 8

class RingBufferQueue{
	public:
		RingBufferQueue();
		~RingBufferQueue();

		void push(const cv::Mat& frame);
		bool pop(cv::Mat& frame);

	private:
		cv::Mat m_ringBuffer[BUFFER_SIZE];
		std::size_t m_producer, m_consumer, m_count;
		std::mutex m_mutex;
};
