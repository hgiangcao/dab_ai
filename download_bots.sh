#!/bin/bash

#!/bin/bash

set -e

mkdir -p ref_bots
cd ref_bots

git clone https://github.com/MarkusThill/DotsAndBoxes.git markus
git clone https://github.com/Camm66/Dots-And-Boxes.git camm66
git clone https://github.com/jnmbk/dotsandboxes.git jnmbk
git clone https://github.com/salil03/dots-and-boxes.git salil03
git clone https://github.com/yimmon/dots-and-boxes.git yimmon
git clone https://github.com/HuXin0817/dots-and-boxes.git huxin

echo
echo "Done."
echo "Repositories downloaded to:"
pwd