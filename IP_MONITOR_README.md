# IP Address Monitoring System

## Overview
The IP Address Monitoring System is a set of scripts that monitor the public IP address of the server and send notifications when it changes. This is useful for servers with dynamic IP addresses where you need to know when the IP changes.

## Components

### 1. IP Check Script (`check_ip_change.sh`)
- Checks the current public IP address using multiple services (ipify.org, ifconfig.me, ipinfo.io)
- Compares it with the previously saved IP address
- If the IP has changed, it creates a notification file

### 2. Email Notification Script (`send_ip_notification.sh`)
- Reads the notification file created by the IP check script
- Sends an email notification with the IP change details using Mailgun API
- Removes the notification file after sending

### 3. Combined Monitoring Script (`monitor_ip_changes.sh`)
- Runs the IP check script
- If a notification file was created, runs the email notification script
- Designed to be run as a cron job

## Files
- `check_ip_change.sh`: Script to check for IP changes
- `send_ip_notification.sh`: Script to send email notifications
- `monitor_ip_changes.sh`: Combined script for cron jobs
- `mailgun_config.sh`: Configuration file for Mailgun credentials
- `current_ip.txt`: File containing the current IP address
- `ip_change.log`: Log file for IP changes
- `ip_changed_notification.txt`: Temporary file created when an IP change is detected
- `ip_cron.log`: Log file for cron job output

## Usage

### Manual Check
To manually check for IP changes:
```bash
./check_ip_change.sh
```

### Manual Notification
To manually send a notification (if a change was detected):
```bash
./send_ip_notification.sh
```

### Combined Check and Notification
To check for changes and send a notification if needed:
```bash
./monitor_ip_changes.sh
```

### Cron Job
The system is set up to run automatically every hour via a cron job:
```
0 * * * * /home/tdeshane/movie_maker/monitor_ip_changes.sh >> /home/tdeshane/movie_maker/ip_cron.log 2>&1
```

## Email Configuration
The system uses Mailgun to send reliable email notifications. To configure Mailgun:

1. Sign up for a Mailgun account at https://www.mailgun.com/
2. Verify your domain or use the sandbox domain provided by Mailgun
3. Get your API key from the Mailgun dashboard
4. Edit the `mailgun_config.sh` file with your credentials:

```bash
# Mailgun configuration
MAILGUN_API_KEY="your-api-key"
MAILGUN_DOMAIN="your-domain.com"
FROM_EMAIL="ip-monitor@your-domain.com"
TO_EMAIL="your-email@example.com"
```

### Mailgun Benefits
- High deliverability rates
- Proper email authentication (SPF, DKIM)
- Less likely to be marked as spam
- Reliable delivery tracking
- Free tier available for low-volume usage

## Logs
- Check `ip_change.log` for a history of IP checks and changes
- Check `ip_cron.log` for the output of the cron job 