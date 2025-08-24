apt update && apt upgrade --yes && apt install vim --yes

echo >>~/.bashrc
echo "set -o vi" >>~/.bashrc
