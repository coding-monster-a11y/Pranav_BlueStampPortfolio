# Pose Estimation
I built a pose-tracking device using a Raspberry Pi, a camera, and a machine learning model that detects key points on a person's body — like shoulders, elbows, and knees — in real time. I fixed several hardware issues along the way, including switching to a different camera library for better reliability, and added a small screen that shows live stats like frame rate and how many body points are being detected. Now the device gives instant visual feedback on screen instead of only saving photos you'd have to check later.

You should comment out all portions of your portfolio that you have not completed yet, as well as any instructions:
```HTML
<!--- This is an HTML comment in Markdown -->
<!--- Anything between these symbols will not render on the published site -->
```

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Pranav G | Irvington High School | Data Science / CSE | Incoming Senior

**Replace the BlueStamp logo below with an image of yourself and your completed project. Follow the guide [here](https://tomcam.github.io/least-github-pages/adding-images-github-pages-site.html) if you need help.**

![Headstone Image](logo.svg)
  
# Final Milestone

**Don't forget to replace the text below with the embedding for your milestone video. Go to Youtube, click Share -> Embed, and copy and paste the code to replace what's below.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/eS6Krnuh100?si=k-Y2pik1kJ2L1Zvb" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For my final milestone, I added a small OLED display to the project, giving it real-time visual feedback instead of only saving results silently to disk. This involved wiring the screen over I2C, troubleshooting a bent GPIO pin, and writing code to show live stats — frame rate, number of body keypoints detected, and system status — directly on the device as it runs. The final result is a fully working, self-contained pose-tracking device that lets you see exactly what it's detecting in real time, right on the hardware itself.


# First Milestone

**Don't forget to replace the text below with the embedding for your milestone video. Go to Youtube, click Share -> Embed, and copy and paste the code to replace what's below.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/hFuY0ifRM_0?si=FQZ-E5-gd6A0W1EU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For my first milestone, I got the core pose estimation system running on the Raspberry Pi, using a TensorFlow Lite PoseNet model to detect body keypoints from a live camera feed. Along the way, I debugged a camera compatibility issue by switching from OpenCV's video capture to the Picamera2 library, then fixed a resulting color tint bug to get accurate images. By the end of this milestone, the system could reliably capture frames, detect body keypoints, and save annotated images to the Pi.

# Schematics 
Here's where you'll put images of your schematics. [Tinkercad](https://www.tinkercad.com/blog/official-guide-to-tinkercad-circuits) and [Fritzing](https://fritzing.org/learning/) are both great resoruces to create professional schematic diagrams, though BSE recommends Tinkercad becuase it can be done easily and for free in the browser. 

# Code
Here's where you'll put your code. The syntax below places it into a block of code. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize it to your project needs. 

```c++
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.println("Hello World!");
}

void loop() {
  // put your main code here, to run repeatedly:

}
```

# Bill of Materials
Here's where you'll list the parts in your project. To add more rows, just copy and paste the example rows below.
Don't forget to place the link of where to buy each component inside the quotation marks in the corresponding row after href =. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize this to your project needs. 

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| Raspberry Pi 4 Starter Kit | Computing | $149 | <a href="[https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6](https://www.amazon.com/RasTech-Raspberry-Starter-Heatsink-Screwdriver/dp/B0C8LV6VNZ/ref=sr_1_3?crid=2LSCXAYVEND7R&dib=eyJ2IjoiMSJ9.VmzHyLmBoYHzxIFtnWbp5P46ovZnz40k9gjFe9rZ0RcEPpkipFreTI-HormqxLXDApvl8921zFN7_RtLChz7gvjkAvd9sXryp5u32gi9jMPsAGx_PqW7CV2z4C925pBi8c-ndO-FdMupX-mTm-I5KEJEg42FwWlXnMK-5XkA5qE1gZfvv7epVYjHd8dpfFX2qpJrR8IOuapFCWP1vfGFMXbDdNlS1i558lxPGB8BISQ.9SNnBJRO2-iAf6iXGLRa-bfHFQpR68AyJDooPtHtiiE&dib_tag=se&keywords=rastech%2B4gb%2Bram&qid=1782846833&sprefix=rastech%2B4gb%2Bram%2Caps%2C179&sr=8-3&th=1)"> Link </a> |
| Item Name | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |
| Item Name | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |

# Other Resources/Examples
One of the best parts about Github is that you can view how other people set up their own work. Here are some past BSE portfolios that are awesome examples. You can view how they set up their portfolio, and you can view their index.md files to understand how they implemented different portfolio components.
- [Example 1](https://trashytuber.github.io/YimingJiaBlueStamp/)
- [Example 2](https://sviatil0.github.io/Sviatoslav_BSE/)
- [Example 3](https://arneshkumar.github.io/arneshbluestamp/)

To watch the BSE tutorial on how to create a portfolio, click here.
