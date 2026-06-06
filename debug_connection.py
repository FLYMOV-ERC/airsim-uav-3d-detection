#!/usr/bin/env python3
import socket
import subprocess
import time

print("🔍 DEBUG DETALHADO DE CONEXÃO\n")

# Pega IP do Windows
result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
host_ip = result.stdout.split('\n')[0].split()[2]
print(f"IP Windows: {host_ip}\n")

# Testa ping
print("1. Testando ping para o Windows:")
ping_result = subprocess.run(['ping', '-c', '2', host_ip], capture_output=True, text=True)
if "0% packet loss" in ping_result.stdout:
    print("   ✅ Windows acessível via ping\n")
else:
    print("   ❌ Problema de conectividade básica\n")

# Testa telnet/nc
print("2. Testando conexão TCP na porta 41451:")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(3)
try:
    result = sock.connect((host_ip, 41451))
    print(f"   ✅ Conectou na porta 41451!")
except socket.timeout:
    print(f"   ❌ Timeout - porta não responde")
except socket.error as e:
    print(f"   ❌ Erro: {e}")
finally:
    sock.close()

print("\n3. Possíveis problemas:")
print("   a) AirSim não está escutando em 0.0.0.0")
print("   b) Windows Defender está bloqueando")
print("   c) Settings.json não foi salvo corretamente")
print("   d) AirSim está usando outra porta")

print("\n4. AÇÕES NO WINDOWS:")
print("   - Execute no PowerShell: netstat -an | findstr LISTENING | findstr 41451")
print("   - Se não aparecer nada, o AirSim NÃO está escutando")
print("   - Verifique o console do AirSim por mensagens de erro")
print("   - Confirme que o settings.json está em: C:\\Users\\[seu_usuario]\\Documents\\AirSim\\settings.json")