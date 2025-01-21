const button = document.querySelector("button");

async function processVideo() {
  // Get the video file from the form
  const videoInput = document.getElementById("video");
  const videoFile = videoInput.files[0];

  if (!videoFile) {
    alert("Please select a video file.");
    return;
  }

  const formData = new FormData();
  formData.append("video", videoFile);

  try {
    // Send the video file to the backend API
    const response = await fetch("/process", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      // Get the processed video as a blob
      const videoBlob = await response.blob();

      // Get the video player element
      const videoPlayer = document.getElementById("processedVideo");

      // Check if the video blob is valid
      if (videoBlob.size === 0) {
        alert("The video is empty or corrupted.");
        return;
      }

      // Create an object URL for the video blob and set it as the source
      const videoURL = URL.createObjectURL(videoBlob);
      videoPlayer.src = videoURL;

      // Show the video player
      videoPlayer.style.display = "block";

      // Ensure the video can start playing when loaded
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

// Make sure to trigger processVideo on button click
button.addEventListener("click", processVideo);
