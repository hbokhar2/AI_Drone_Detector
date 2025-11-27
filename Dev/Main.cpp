#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"

#include "CameraProducer/CameraProducer.h"
#include "RingBufferQueue/RingBufferQueue.h"
#include "YoloProcessor/YoloProcessor.h"

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

	std::unique_ptr<YoloProcessor> processor = nullptr;

	try {
		std::cout << "Attempting to initialize AI Processor..." << std::endl;
		processor = std::make_unique<YoloProcessor>(
				"", 
				#ifdef __linux__
				"/home/B0LD/Documents/Projects/Capstone/DroneDetection/AiTrainer/TrainedAiFiles/drone_run/weights/best.onnx", 
				"/home/B0LD/Documents/Projects/Capstone/DroneDetection/drone_dataset/drone.names"
				#elif defined(_WIN32) || defined(_WIN64)
				//Most people will compile for windows (likely) so add windows paths accordingly.
				#endif
				);
		std::cout << "AI Processor successfully initialized YOLO model." << std::endl;

	} catch (const std::exception& e) {
		std::cerr << "AI PROCESSOR FAILED TO INITIALIZE: " << e.what() << std::endl;
		return; 
	}

	cv::Mat rawFrame;

	while(true){
		if (cam_to_ai_raw.pop(rawFrame)) {
			if (rawFrame.empty()) continue; 

			cv::Mat annotatedFrame = processor->process_frame(rawFrame); 

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
