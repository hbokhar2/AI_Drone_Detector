#include "RingBufferQueue.h"
#include <mutex>

RingBufferQueue::RingBufferQueue(){
	m_producer = 0;
	m_consumer = 0;
	m_count = 0;
}

RingBufferQueue::~RingBufferQueue(){
}

void RingBufferQueue::push(const cv::Mat& frame){
	std::lock_guard<std::mutex> lock(m_mutex);

	if(m_count == BUFFER_SIZE){
		m_consumer = (m_consumer + 1) % BUFFER_SIZE;
		m_count--;
	}

	m_ringBuffer[m_producer] = frame;
	m_producer = (m_producer + 1) % BUFFER_SIZE;
	m_count++;
}

bool RingBufferQueue::pop(cv::Mat& frame){
	std::lock_guard<std::mutex> lock(m_mutex);

	if(m_count == 0) return false;

	frame = m_ringBuffer[m_consumer];

	m_consumer = (m_consumer + 1) % BUFFER_SIZE;
	m_count--;

	return true;
}
