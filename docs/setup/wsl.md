# Connecting AirSim (Windows) to a WSL client

> **Note.** Headings and titles in this document have been translated to English;
> the body text is still the original Portuguese. It is published as a working
> record, not as a polished English document.
>
> **Status.** WSL ↔ Windows networking. The host address is no longer hardcoded
> anywhere: set `airsim.host` in `configs/default.yaml`, or export `AIRSIM_HOST`.
> The connection test script this document referred to was removed; a one-line
> check is `python -c "import cosysairsim as a; c=a.MultirotorClient(ip='<host>'); c.confirmConnection()"`.


## PROBLEMA DETECTADO
O AirSim no Windows não está aceitando conexões do WSL.

## Step-by-step solution

### 1. On Windows: configure settings.json

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

**IMPORTANTE**: As linhas `"LocalHostIp": "0.0.0.0"` e `"ApiServerPort": 41451` são ESSENCIAIS!

### 2. On Windows: configure the firewall

Abra o **PowerShell como Administrador** e execute:

```powershell
New-NetFirewallRule -DisplayName "AirSim WSL" -Direction Inbound -Protocol TCP -LocalPort 41451 -Action Allow
```

### 3. On Windows: restart AirSim

1. **FECHE** completamente o simulador (Blocks.exe ou AirSimNH.exe)
2. **ABRA** novamente
3. **AGUARDE** carregar completamente (até aparecer o drone)

### 4. In WSL: test the connection

Execute:
```bash
python3 test_connection_advanced.py
```

## Verification on Windows

Para confirmar que o AirSim está escutando, no PowerShell:

```powershell
netstat -an | findstr 41451
```

Você deve ver algo como:
```
TCP    0.0.0.0:41451    0.0.0.0:0    LISTENING
```

## If it still does not work

1. **Desative temporariamente o Windows Defender Firewall** (apenas para teste!)
2. **Verifique se você salvou o settings.json corretamente**
3. **Certifique-se de que reiniciou o AirSim após editar o settings.json**
4. **Tente com porta diferente** (ex: 9000) em ambos os arquivos

## Example of a working connection

Quando funcionar, você verá:
```
CONECTADO COM SUCESSO!
Veículos disponíveis: ['Ego']
Veículo: Ego
   Posição: X=0.00, Y=0.00, Z=-2.00
```