# 🚁 CONECTAR AIRSIM (WINDOWS) COM WSL

## ❌ PROBLEMA DETECTADO
O AirSim no Windows não está aceitando conexões do WSL.

## ✅ SOLUÇÃO PASSO A PASSO

### 1️⃣ NO WINDOWS: Configure o settings.json

1. Abra o Windows Explorer
2. Navegue até: `C:\Users\[seu_usuario]\Documents\AirSim\`
3. Abra o arquivo `settings.json` (crie se não existir)
4. **SUBSTITUA TODO O CONTEÚDO** por:

```json
{
  "SettingsVersion": 1.2,
  "LocalHostIp": "0.0.0.0",
  "ApiServerPort": 41451,
  "SimMode": "Multirotor",
  "ClockSpeed": 1.0,
  "ViewMode": "SpringArmChase",
  "Vehicles": {
    "Ego": {
      "VehicleType": "SimpleFlight",
      "X": 0,
      "Y": 0,
      "Z": -2,
      "Yaw": 0
    }
  }
}
```

⚠️ **IMPORTANTE**: As linhas `"LocalHostIp": "0.0.0.0"` e `"ApiServerPort": 41451` são ESSENCIAIS!

### 2️⃣ NO WINDOWS: Configure o Firewall

Abra o **PowerShell como Administrador** e execute:

```powershell
New-NetFirewallRule -DisplayName "AirSim WSL" -Direction Inbound -Protocol TCP -LocalPort 41451 -Action Allow
```

### 3️⃣ NO WINDOWS: Reinicie o AirSim

1. **FECHE** completamente o simulador (Blocks.exe ou AirSimNH.exe)
2. **ABRA** novamente
3. **AGUARDE** carregar completamente (até aparecer o drone)

### 4️⃣ NO WSL: Teste a conexão

Execute:
```bash
python3 test_connection_advanced.py
```

## 🔍 VERIFICAÇÃO NO WINDOWS

Para confirmar que o AirSim está escutando, no PowerShell:

```powershell
netstat -an | findstr 41451
```

Você deve ver algo como:
```
TCP    0.0.0.0:41451    0.0.0.0:0    LISTENING
```

## 🆘 SE AINDA NÃO FUNCIONAR

1. **Desative temporariamente o Windows Defender Firewall** (apenas para teste!)
2. **Verifique se você salvou o settings.json corretamente**
3. **Certifique-se de que reiniciou o AirSim após editar o settings.json**
4. **Tente com porta diferente** (ex: 9000) em ambos os arquivos

## 📝 EXEMPLO DE CONEXÃO FUNCIONANDO

Quando funcionar, você verá:
```
✅ CONECTADO COM SUCESSO!
🚁 Veículos disponíveis: ['Ego']
📍 Veículo: Ego
   Posição: X=0.00, Y=0.00, Z=-2.00
```