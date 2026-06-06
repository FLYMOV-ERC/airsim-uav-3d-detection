# test_conn.py
try:
    import airsim  # muitos builds do cosysairsim expõem o módulo como 'airsim'
except Exception:
    import cosysairsim as airsim

c = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
c.confirmConnection()
print("OK conectado!")
