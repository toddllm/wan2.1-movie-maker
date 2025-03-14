#!/bin/bash
# Combined script to check for IP changes and send notifications
# This script is designed to be run as a cron job

# Change to the movie_maker directory
cd "$HOME/movie_maker"

# Run the IP check script
./check_ip_change.sh

# If a notification file was created, send the notification
if [ -f "$HOME/movie_maker/ip_changed_notification.txt" ]; then
    ./send_ip_notification.sh
fi

exit 0 