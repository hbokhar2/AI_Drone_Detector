#include <string>

#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>

class CameraProducer{
	public:
		CameraProducer();
		~CameraProducer();

		void initialize();
		cv::Mat produce_frames();

	private:

		//std::string m_pipeline = "v4l2src device=/dev/video0 ! image/jpeg, width=1600, height=600, framerate=60/1 ! jpegdec ! videoconvert ! appsink";
		std::string m_pipeline = "v4l2src device=/dev/video0 ! image/jpeg, width=640, height=240, framerate=60/1 ! jpegdec ! videoconvert ! appsink";
		cv::VideoCapture m_videoCap;
		cv::Mat m_rawFrames;
};

