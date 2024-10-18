# 7-DOF Robotic Arm Simulation with Gymnasium and Mujoco

This project simulates a 7-degree-of-freedom (7-DOF) robotic arm using the Gymnasium framework and the Mujoco physics engine. The setup includes a customized push task environment, where the robotic arm interacts with a golf ball and a target.

## Requirements

To run this project, ensure you have the following software installed:

- Python version 3.9 or above

### Libraries Used

The following libraries are required:

- **Gymnasium + Gymnasium Robotics**: For environment creation and interaction with agents.
- **Mujoco**: For physics-based simulations involving robotics.
- **NumPy**: For numerical operations and data management.

#### Running the Project

To run the project :
- Ensure all dependencies are installed.
- Navigate to the directory containing the Python files.
- Run the setup script to register the environment and run the          simulation for 1000 steps with random actions.


Install the necessary libraries using pip:

```bash
pip install gymnasium[robotics] mujoco numpy

