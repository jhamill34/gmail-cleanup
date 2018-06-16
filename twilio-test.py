from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = "ACb7a4001997135d639d2cd5f6fb548e3d"
# Your Auth Token from twilio.com/console
auth_token  = ""

client = Client(account_sid, auth_token)

message = client.messages.create(
    to="+14257609918",
    from_="+14252305280",
    body="Hello from Python!")

print(message.sid)
