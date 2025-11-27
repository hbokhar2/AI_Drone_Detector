#include "YoloProcessor.h"

#include <fstream>
#include <iostream>

#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgproc.hpp"

void YoloProcessor::load_class_names(const std::string& namesPath) {
	std::ifstream ifs(namesPath.c_str());
	if (ifs.is_open()) {
		std::string line;
		while (std::getline(ifs, line)) {
			m_classNames.push_back(line);
		}
		std::cout << "Loaded " << m_classNames.size() << " class names from " << namesPath << "." << std::endl;
	} else {
		std::cerr << "CRITICAL ERROR: Could not open names file: " << namesPath << std::endl;
		throw std::runtime_error("Names file missing.");
	}
}

std::vector<std::string> YoloProcessor::get_output_layer_names(const cv::dnn::Net& net) {
	std::vector<std::string> names;
	std::vector<int> outLayers = net.getUnconnectedOutLayers();
	std::vector<std::string> layersNames = net.getLayerNames();

	for (int i : outLayers) {
		names.push_back(layersNames[i - 1]);
	}
	return names;
}

YoloProcessor::YoloProcessor(const std::string& configPath, 
		const std::string& weightsPath, 
		const std::string& namesPath) 
{
	load_class_names(namesPath);

	try {
		m_net = cv::dnn::readNetFromONNX(weightsPath); 

		m_outputLayerNames = get_output_layer_names(m_net);

		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	} catch (const cv::Exception& e) {
		std::cerr << "OpenCV DNN Error loading model: " << e.what() << std::endl;
		throw;
	}
}

cv::Mat YoloProcessor::process_frame(const cv::Mat& inputFrame){
	if(inputFrame.empty()) return inputFrame;

	cv::Mat annotatedFrame = inputFrame.clone();

	cv::Mat blob;
	cv::dnn::blobFromImage(inputFrame, blob, 1/255.0, cv::Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), 
			cv::Scalar(), true, false);

	m_net.setInput(blob);
	std::vector<cv::Mat> detections;
	m_net.forward(detections, m_outputLayerNames);

	post_process(annotatedFrame, detections);

	return annotatedFrame;
}

void YoloProcessor::post_process(cv::Mat& frame, const std::vector<cv::Mat>& outputs) {
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	cv::Mat output = outputs[0];

	int prediction_dimensions = output.size[1]; 

	output = output.reshape(0, prediction_dimensions); 

	cv::transpose(output, output); 

	int num_predictions = output.rows;
	int num_data_points = output.cols;

	float* data = (float*)output.data;


	for (int i = 0; i < num_predictions; ++i) {
		float confidence = data[4]; 

		if (confidence >= CONFIDENCE_THRESHOLD) {

			float cx = data[0];
			float cy = data[1];
			float w = data[2];
			float h = data[3];

			const float YOLO_INPUT_SIZE = 416.0f;
			float scale_w = (float)frame.cols / YOLO_INPUT_SIZE;
			float scale_h = (float)frame.rows / YOLO_INPUT_SIZE;

			int left = (int)((cx - 0.5 * w) * scale_w);
			int top = (int)((cy - 0.5 * h) * scale_h);
			int width = (int)(w * scale_w);
			int height = (int)(h * scale_h);

			class_ids.push_back(0);
			confidences.push_back(confidence);
			boxes.push_back(cv::Rect(left, top, width, height));
		}

		data += num_data_points; 
	}


	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

	for (int idx : indices) {
		cv::Rect box = boxes[idx];

		cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

		int classId = class_ids[idx];
		std::string label = (m_classNames.empty() || classId >= m_classNames.size()) ? "Drone" : m_classNames[classId]; 
		label += cv::format(" (%.2f)", confidences[idx]);

		cv::putText(frame, label, cv::Point(box.x, box.y - 10), 
				cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
	}
}
