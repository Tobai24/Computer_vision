const button = document.querySelector("button");

async function processVideo() {
  const videoInput = document.getElementById("video");
  const videoFile = videoInput.files[0];

  if (!videoFile) {
    alert("Please select a video file.");
    return;
  }

  const formData = new FormData();
  formData.append("video", videoFile);

  try {
    const response = await fetch("/process", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const videoBlob = await response.blob();
      const videoPlayer = document.getElementById("processedVideo");

      // Check if the video blob is valid
      if (videoBlob.size === 0) {
        alert("The video is empty or corrupted.");
        return;
      }

      const videoURL = URL.createObjectURL(videoBlob);
      videoPlayer.src = videoURL;
      videoPlayer.style.display = "block";
      videoPlayer.load();
      videoPlayer.play();
    } else {
      alert("Failed to process video.");
    }
  } catch (error) {
    console.error("Error during video processing:", error);
    alert("An error occurred while processing the video.");
  }
}

button.addEventListener("click", processVideo);
