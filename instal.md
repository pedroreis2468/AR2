
---

## Building PacSim on Ubuntu 24.04 LTS (ROS 2 Jazzy)

PacSim was originally designed for Ubuntu 22.04 (ROS 2 Iron). However, it can generally be built on Ubuntu 24.04 (ROS 2 Jazzy) provided your Python environment is set up correctly.

**Important Note for Conda Users:** Building ROS 2 packages while an Anaconda/Miniconda environment (like `(base)`) is active will cause CMake to use Conda's isolated Python instead of the system Python. This leads to missing module errors (e.g., `ModuleNotFoundError: No module named 'em'`).

### Step 1: Deactivate Conda

Before doing anything with ROS 2, ensure you are out of any Conda environments so CMake uses the default system Python.

```bash
conda deactivate

```

*(Ensure the `(base)` prefix disappears from your terminal prompt).*

### Step 2: Install Dependencies

Install ROS 2 Jazzy (if you haven't already), the build tools, and the necessary system-level Python packages that ROS 2 relies on for generating interfaces.

```bash
sudo apt update
sudo apt install ros-jazzy-desktop ros-jazzy-xacro python3-colcon-common-extensions python3-empy python3-catkin-pkg -y

```

### Step 3: Create Workspace and Clone

Set up your ROS 2 workspace directory and pull the PacSim source code from GitHub.

```bash
mkdir -p ~/pacsim_ws/src
cd ~/pacsim_ws/src
git clone https://github.com/PacSim/pacsim.git

```

### Step 4: Clean the Workspace

If you previously attempted a build that failed (especially while Conda was active), CMake will have cached the wrong paths. You must clear these out before trying again.

```bash
cd ~/pacsim_ws
rm -rf build/ install/ log/

```

### Step 5: Source ROS 2 and Build

Source your global ROS 2 Jazzy installation to make the build tools available, then compile the package.

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install

```

### Step 6: Run the Simulator

Once the build finishes successfully, source your local workspace overlay and launch the example file to verify it works.

```bash
source install/setup.bash
ros2 launch pacsim example.launch.py

```

---

Would you like me to help you set up the Foxglove Studio visualization next, or did the `colcon build` command throw any new C++ warnings when compiling against Jazzy?
