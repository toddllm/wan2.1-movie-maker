#!/bin/bash
# Script to send an email notification when the IP address changes
# Uses Mailgun API to send reliable emails

# Configuration
NOTIFICATION_FILE="$HOME/movie_maker/ip_changed_notification.txt"
LOG_FILE="$HOME/movie_maker/ip_change.log"
CONFIG_FILE="$HOME/movie_maker/mailgun_config.sh"

# Mailgun configuration (will be loaded from config file)
MAILGUN_API_KEY=""
MAILGUN_DOMAIN=""
FROM_EMAIL=""
TO_EMAIL=""
EMAIL_SUBJECT="Server IP Address Change Notification"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    echo "$1"
}

# Check if notification file exists
if [ ! -f "$NOTIFICATION_FILE" ]; then
    log_message "No IP change notification file found"
    exit 0
fi

# Read the notification content
NOTIFICATION_CONTENT=$(cat "$NOTIFICATION_FILE")

# Check if config file exists and load it
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    log_message "Loaded Mailgun configuration from $CONFIG_FILE"
else
    log_message "ERROR: Mailgun configuration file not found at $CONFIG_FILE"
    log_message "Please create this file with the following content:"
    log_message "MAILGUN_API_KEY=\"your-api-key\""
    log_message "MAILGUN_DOMAIN=\"your-domain.com\""
    log_message "FROM_EMAIL=\"ip-monitor@your-domain.com\""
    log_message "TO_EMAIL=\"your-email@example.com\""
    
    # Create a template config file
    echo '# Mailgun configuration
MAILGUN_API_KEY="your-api-key"
MAILGUN_DOMAIN="your-domain.com"
FROM_EMAIL="ip-monitor@your-domain.com"
TO_EMAIL="your-email@example.com"
' > "$CONFIG_FILE"
    
    log_message "Created template configuration file at $CONFIG_FILE"
    log_message "Please edit this file with your Mailgun credentials"
    
    # Still log the notification content
    log_message "Would send email with content: $NOTIFICATION_CONTENT"
    exit 1
fi

# Validate configuration
if [ -z "$MAILGUN_API_KEY" ] || [ -z "$MAILGUN_DOMAIN" ] || [ -z "$FROM_EMAIL" ] || [ -z "$TO_EMAIL" ]; then
    log_message "ERROR: Incomplete Mailgun configuration. Please check $CONFIG_FILE"
    exit 1
fi

# Log that we're sending an email
log_message "Sending email notification via Mailgun: $NOTIFICATION_CONTENT"

# Send email using Mailgun API
RESPONSE=$(curl -s --user "api:$MAILGUN_API_KEY" \
    "https://api.mailgun.net/v3/$MAILGUN_DOMAIN/messages" \
    -F from="IP Monitor <$FROM_EMAIL>" \
    -F to="$TO_EMAIL" \
    -F subject="$EMAIL_SUBJECT" \
    -F text="$NOTIFICATION_CONTENT")

# Check if the email was sent successfully
if [[ "$RESPONSE" == *"Queued"* ]]; then
    log_message "Email sent successfully via Mailgun"
else
    log_message "ERROR: Failed to send email via Mailgun. Response: $RESPONSE"
fi

# Remove the notification file after sending
rm "$NOTIFICATION_FILE"
log_message "Notification file removed after processing" 