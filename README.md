# eyecontrol
opencv eye controlling
This project implements an eye-controlled interface that allows users to interact with their computer using eye movements and blinks. It leverages the MediaPipe Face Mesh library to accurately detect and track facial landmarks, specifically focusing on the eyes.

Here's a breakdown of its core functionalities:

Blink Detection for Actions:

It calculates the Eye Aspect Ratio (EAR) for both eyes to determine if a blink has occurred.
A single blink triggers a "select" action.
A double blink triggers a "confirm" action.
A long blink (holding the blink for a specified duration) acts as a "cancel" mechanism, preventing accidental actions.
Vertical Eye Movement for Scrolling:

The system tracks the vertical position of the eye center.
By analyzing the change in this position over a short history, it determines if the user is looking up or down.
This movement is then translated into scrolling actions (up or down) using the pyautogui library, allowing hands-free navigation of documents or web pages.
Essentially, this project offers an alternative input method for users, potentially benefiting those with limited mobility or as a novel way to interact with a computer.
