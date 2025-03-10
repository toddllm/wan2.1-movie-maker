#!/bin/bash
# Deploy Movie Maker as a systemd service

# Exit on any error
set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

# Define paths
SERVICE_NAME="movie-maker"
SERVICE_FILE="$SERVICE_NAME.service"
CURRENT_DIR=$(dirname "$(readlink -f "$0")")
SYSTEMD_DIR="/etc/systemd/system"

echo "Deploying Movie Maker as a systemd service..."

# Make run.sh executable
chmod +x "$CURRENT_DIR/run.sh"

# Ensure the clips and movies directories exist
mkdir -p "$CURRENT_DIR/clips" "$CURRENT_DIR/movies"
chown -R "$(logname)":"$(logname)" "$CURRENT_DIR/clips" "$CURRENT_DIR/movies"

# Copy the service file to systemd directory
echo "Installing service file to $SYSTEMD_DIR/$SERVICE_FILE"
cp "$CURRENT_DIR/$SERVICE_FILE" "$SYSTEMD_DIR/$SERVICE_FILE"

# Update the service file with the correct username if needed
CURRENT_USER=$(logname)
if [ "$CURRENT_USER" != "tdeshane" ]; then
    echo "Updating service file to use the current user: $CURRENT_USER"
    sed -i "s/User=tdeshane/User=$CURRENT_USER/g" "$SYSTEMD_DIR/$SERVICE_FILE"
    HOME_DIR=$(eval echo ~$CURRENT_USER)
    sed -i "s|WorkingDirectory=/home/tdeshane/movie_maker|WorkingDirectory=$HOME_DIR/movie_maker|g" "$SYSTEMD_DIR/$SERVICE_FILE"
    sed -i "s|ExecStart=/home/tdeshane/movie_maker/run.sh|ExecStart=$HOME_DIR/movie_maker/run.sh|g" "$SYSTEMD_DIR/$SERVICE_FILE"
fi

# Reload systemd to recognize the new service
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable and start the service
echo "Enabling and starting the service..."
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"

# Check status
echo "Service status:"
systemctl status "$SERVICE_NAME" --no-pager

echo ""
echo "Movie Maker service has been deployed!"
echo "You can access it at: http://$(hostname -I | awk '{print $1}'):5001"
echo ""
echo "Useful commands:"
echo "- Check service status: sudo systemctl status $SERVICE_NAME"
echo "- Stop the service: sudo systemctl stop $SERVICE_NAME"
echo "- Start the service: sudo systemctl start $SERVICE_NAME"
echo "- Restart the service: sudo systemctl restart $SERVICE_NAME"
echo "- View logs: sudo journalctl -u $SERVICE_NAME -f" 