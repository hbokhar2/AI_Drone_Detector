#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"

#include "CameraProducer/CameraProducer.h"
#include "RingBufferQueue/RingBufferQueue.h"

void camera_producer_thread(CameraProducer& producer, RingBufferQueue& cam_to_ai_raw){
	producer.initialize();
	while(true){
		cv::Mat frame = producer.produce_frames();
		if(!frame.empty()){
			cam_to_ai_raw.push(frame);
		}else{
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}
	}
}

void annotated_display_thread(RingBufferQueue& ai_to_display_annotated){
	cv::namedWindow("Annotated AI Output");
	cv::Mat frame;
	bool success;
	while(cv::waitKey(1) != 27){
		success = ai_to_display_annotated.pop(frame);
		if(success){
			cv::imshow("Annotated AI Output", frame);
		}else{
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}
	}
	cv::destroyAllWindows();
}

void ai_processor_thread(RingBufferQueue& cam_to_ai_raw, RingBufferQueue& ai_to_display_annotated){
	cv::Mat rawFrame;

	while(true){
		if (cam_to_ai_raw.pop(rawFrame)) {
			if (rawFrame.empty()) continue; 

			cv::Mat annotatedFrame = rawFrame.clone(); 
			//Put Yolo processing code here.

			ai_to_display_annotated.push(annotatedFrame);

		} else {
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}
	}
}

int main(){
	CameraProducer cameraProducer;
	RingBufferQueue cam_to_ai, ai_to_display;

	std::cout << "Launching Producer, AI Processor, and Display threads..." << std::endl;

	std::thread t1(camera_producer_thread, 
			std::ref(cameraProducer), 
			std::ref(cam_to_ai));

	std::thread t2(annotated_display_thread, 
			std::ref(ai_to_display));

	std::thread t3(ai_processor_thread, 
			std::ref(cam_to_ai), 
			std::ref(ai_to_display));

	std::cout << "OpenCV GUI running in Consumer thread. Close the window or press ESC to exit." << std::endl;

	t2.join();

	if (t1.joinable()) {
		std::cout << "Detaching Producer thread." << std::endl;
		t1.detach();
	}
	if (t3.joinable()) {
		std::cout << "Detaching AI Processor thread." << std::endl;
		t3.detach();
	}

	return 0;
}
