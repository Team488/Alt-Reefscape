# Alt
The only way to get a map through the Blackwall

## Getting Started

Run in linux (use wsl2 to install Ubuntu on Windows).

### Windows

```powershell
wsl.exe --install --d Ubuntu-22.04
```

Then open a new terminal that is running Ubuntu (In Windows Terminal or VS Code there is a + sign).

#### Installing VS Code into your Linux running on Windows:
```bash
curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
sudo install -o root -g root -m 644 microsoft.gpg /etc/apt/keyrings/microsoft-archive-keyring.gpg
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/microsoft-archive-keyring.gpg] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
```

Run `sudo apt update` to update the repos then run the following to install vscode and git `sudo apt install git code`.
Then run `ssh-keygen -t ed25519` to generate a new ssh key for cloning github repos. Add the `~/.ssh/id_ed25519.pub` to your github profile's ssh access token.

Now continue with the Linux instructions below.

#### Continuing using Linux

Install pyenv and pyenv virtualenv.

Copy below to install and run the dependencies needed to install pyenv.

```bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev libhdf5-dev
```

Then run the following:

```bash
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
```

```bash
cat <<'EOF' >> $HOME/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

if [[ -x $(command -v pyenv) ]]; then
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi
EOF
```

Run `./setup.sh` afterwards.

### Running your first demo script

At the root of this directory run the following:

```bash
./run.sh src/mapDemos//probmapClickDemo.py
```

## Making a deployable version of Alt

You'll need to be able to emulate running on an ARM based system to build an ARM image for the orange pi.

QEMU will allow for emulation to be able to build for other platforms.

There are caveats here, but this should perform the action that you are looking for.
```bash
docker run --privileged --rm tonistiigi/binfmt --install all
```

The steps are outlined below if you run into issues:
https://docs.docker.com/build/building/multi-platform/#install-qemu-manually

Once that's good you can perform the following steps in order to manually build for the platform.
```bash
docker buildx build --platform linux/arm64 -t alt:arm64 .
docker image save -o alt-arm64.tar alt:arm64
```

## General Notes

Alt ingests NetworkTable inputs of:
* Robot Self Pose in Field Frame
* Robot Frame bboxes of
 * Notes
 * Robots

And generates a probabilistic map in Field Frame of:
* Notes
* Robots
 * Position
 * Probabilistic Trajectory

It also creates a short culled list of:
* Closest three Notes to the robot
* Potential intersecting Robots
