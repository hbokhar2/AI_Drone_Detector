#include <chrono>
#include <iostream>
#include <thread>

#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"

#include "CameraProducer/CameraProducer.h"
#include "RingBufferQueue/RingBufferQueue.h"

void normal_camera_producer_thread(CameraProducer& producer, RingBufferQueue& rb){
	producer.initialize(); 
	while(true){
		cv::Mat frame = producer.produce_frames();
		if(!frame.empty()){
			rb.push(frame);
		}else{
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}
	}
}

void normal_camera_consumer_thread(RingBufferQueue& rb){
	cv::namedWindow("Normal Camera");
	cv::Mat frame;
	bool success;
	while(cv::waitKey(1) != 27){
		success = rb.pop(frame);
		if(success){
			cv::imshow("Normal Camera", frame);
		}else{
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}
	}
	cv::destroyAllWindows();
}

int main(){
	CameraProducer cameraProducer;
	RingBufferQueue ringBufferNormal;

	std::cout << "Launching Producer and Consumer threads..." << std::endl;

	std::thread t1(normal_camera_producer_thread, std::ref(cameraProducer), std::ref(ringBufferNormal));
	std::thread t2(normal_camera_consumer_thread, std::ref(ringBufferNormal));

	std::cout << "OpenCV GUI running in Consumer thread. Close the window or press ESC to exit." << std::endl;

	t2.join(); // Block the main thread until the consumer thread finishes.

	if (t1.joinable()) {
		std::cout << "Detaching Producer thread." << std::endl;
		t1.detach();
	}

	return 0;
}
