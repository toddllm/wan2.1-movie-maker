#!/bin/bash
# Script to check if the public IP address has changed
# If it has changed, it will save the new IP and prepare to send an email notification

# Configuration
IP_FILE="$HOME/movie_maker/current_ip.txt"
LOG_FILE="$HOME/movie_maker/ip_change.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    echo "$1"
}

# Create log file if it doesn't exist
if [ ! -f "$LOG_FILE" ]; then
    touch "$LOG_FILE"
    log_message "Created log file"
fi

# Get the current public IP address
# We'll try multiple services in case one is down
get_public_ip() {
    # Try ipify.org first
    CURRENT_IP=$(curl -s https://api.ipify.org)
    
    # If that fails, try ifconfig.me
    if [ -z "$CURRENT_IP" ]; then
        CURRENT_IP=$(curl -s https://ifconfig.me)
    fi
    
    # If that fails too, try ipinfo.io
    if [ -z "$CURRENT_IP" ]; then
        CURRENT_IP=$(curl -s https://ipinfo.io/ip)
    fi
    
    echo "$CURRENT_IP"
}

# Get the current public IP
CURRENT_IP=$(get_public_ip)

# Check if we got a valid IP
if [ -z "$CURRENT_IP" ]; then
    log_message "ERROR: Could not determine public IP address"
    exit 1
fi

# Check if the IP file exists
if [ ! -f "$IP_FILE" ]; then
    # First run - save the current IP
    echo "$CURRENT_IP" > "$IP_FILE"
    log_message "First run - saved current IP: $CURRENT_IP"
    exit 0
fi

# Read the previous IP
PREVIOUS_IP=$(cat "$IP_FILE")

# Compare IPs
if [ "$CURRENT_IP" != "$PREVIOUS_IP" ]; then
    # IP has changed
    log_message "IP address changed from $PREVIOUS_IP to $CURRENT_IP"
    
    # Save the new IP
    echo "$CURRENT_IP" > "$IP_FILE"
    
    # Prepare email notification (will be implemented later)
    # For now, just create a notification file that can be checked
    echo "IP address changed from $PREVIOUS_IP to $CURRENT_IP at $(date)" > "$HOME/movie_maker/ip_changed_notification.txt"
    
    # In the future, this is where we'll send an email
    # send_email_notification "$PREVIOUS_IP" "$CURRENT_IP"
    
    log_message "Notification prepared for IP change"
    exit 0
else
    # No change
    log_message "IP address unchanged: $CURRENT_IP"
    exit 0
fi 