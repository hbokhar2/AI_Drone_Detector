#include "opencv2/core/mat.hpp"

class CameraConsumer{
	public:
		CameraConsumer();
		~CameraConsumer();

		bool initialize(std::string windowName);
		void update_window();

	private:
		cv::Mat m_frame;
		std::string m_windowName;
};
