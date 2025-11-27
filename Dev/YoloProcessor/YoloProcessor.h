#include <vector>
#include <string> // Added for string usage

#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

class YoloProcessor {
	public:
		YoloProcessor(const std::string& configPath, 
				const std::string& weightsPath, 
				const std::string& namesPath);

		~YoloProcessor() = default; 

		cv::Mat process_frame(const cv::Mat& inputFrame);

	private:
		cv::dnn::Net m_net;
		std::vector<std::string> m_outputLayerNames;
		std::vector<std::string> m_classNames;

		const int YOLO_INPUT_SIZE = 416;
		const float CONFIDENCE_THRESHOLD = 0.5f;
		const float NMS_THRESHOLD = 0.4f;

		std::vector<std::string> get_output_layer_names(const cv::dnn::Net& net);
		void load_class_names(const std::string& namesPath);

		void post_process(cv::Mat& frame, const std::vector<cv::Mat>& outputs); 

};
