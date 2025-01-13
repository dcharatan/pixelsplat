ffmpeg \
    -framerate 30 \
    -i /data/guest_storage/zhanpengluo/FeedForwardGS/pixelsplat/outputs/rollerblade/dynamic/rb4d_2/color/%06d.png \
    -c:v libx264 \
    -pix_fmt yuv420p \
    /data/guest_storage/zhanpengluo/FeedForwardGS/pixelsplat/video/rb4d.mp4
