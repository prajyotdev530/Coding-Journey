def take_bid(bids):
     name=input("what is your name")
     bid_value=input("what is your bid amount")
     bids[name]=bid_value
     bidder=input("are there any other bidder")
     if bidder=="yes":
          
          take_bid(bids)

bids={}
take_bid(bids)
max_bid=0
for key,value in bids.items():
     if int(value)>max_bid:
          max_bid=int(value)
          person=key
print(f"the highest bid is of{person} and the bid is{max_bid}")

