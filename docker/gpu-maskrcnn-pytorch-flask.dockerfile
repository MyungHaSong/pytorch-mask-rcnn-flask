FROM azadehkhojandi/gpu-pytorch-notebook

ENV GPU_Arch=sm_37

USER $NB_UID

USER root
COPY ./clone.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/clone.sh
RUN chmod 777 /home/$NB_USER/work
RUN clone.sh
# ENTRYPOINT ["tini", "-g", "--"]
# CMD ["sh","-c","clone.sh && start-notebook.sh"]

RUN chmod -R 777 /home/$NB_USER/work

RUN pip install flask
RUN mkdir /home/$NB_USER/work/pytorch-mask-rcnn-flask/uploads
RUN cd /home/$NB_USER/work/pytorch-mask-rcnn-flask && \
  python application.py

# Switch back to jovyan to avoid accidental container runs as root
#USER $NB_UID

