FROM tensorflow/tensorflow:latest

RUN apt-get update && \
    apt-get install -y \
        python3.9 \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \

RUN pip3 install --upgrade pip pyyaml
COPY mnist-cnn /mnist-cnn
WORKDIR /mnist-cnn

ENTRYPOINT ["python3"]
CMD [ "train_model.py" ]