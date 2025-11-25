#include "CameraConsumer.h"
#include "opencv2/highgui.hpp"

CameraConsumer::CameraConsumer(){
	m_windowName = "null";
}

CameraConsumer::~CameraConsumer(){
}

bool CameraConsumer::initialize(std::string windowName){
	cv::namedWindow(windowName);
	m_windowName = windowName;
	if(m_windowName == "null") return false;
	return true;
}

void CameraConsumer::update_window(){
	cv::imshow(m_windowName, m_frame);
}
