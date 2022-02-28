docker build --network host -t mbtrex --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f ./Dockerfile .
