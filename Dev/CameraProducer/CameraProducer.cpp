#include <iostream>

#include "CameraProducer.h"

CameraProducer::CameraProducer(){
}

CameraProducer::~CameraProducer(){
}

void CameraProducer::initialize(){
	#ifdef __linux__
		//m_videoCap = cv::VideoCapture{m_pipeline, cv::CAP_GSTREAMER};
	#elif defined(_WIN32) || defined(_WIN64)
		m_videoCap = cv::VideoCapture(0);
	#endif		

	if(!m_videoCap.isOpened()){
		std::cout << "Error opening camera." << std::endl;
		return;
	}
}

cv::Mat CameraProducer::produce_frames(){
	if(!m_videoCap.read(m_rawFrames) || m_rawFrames.empty()){
		std::cerr << "Streaming error, failed to read new frame." << std::endl;
		return cv::Mat();
	}

	return m_rawFrames.clone();
}
