# Configuring AirSim on Windows for a WSL client

> **Note.** Headings and titles in this document have been translated to English;
> the body text is still the original Portuguese. It is published as a working
> record, not as a polished English document.
>
> **Status.** Windows-side AirSim/Unreal setup. The quick-test script this
> document referred to was removed; use the one-line check in docs/setup/wsl.md.


## 1. Check that AirSim is running

No Windows, certifique-se de que um dos seguintes está aberto:
- **Blocks.exe** (ambiente Blocks)
- **AirSimNH.exe** (Neighborhood Environment)
- **Seu projeto Unreal** com plugin AirSim

## 2. Configure settings.json on Windows

Localize o arquivo em: `C:\Users\[seu_usuario]\Documents\AirSim\settings.json`

Adicione estas configurações para permitir conexões externas:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "LocalHostIp": "0.0.0.0",
  "ApiServerPort": 41451,
  // ... resto das suas configurações
}
```

**IMPORTANTE**: Adicione `"LocalHostIp": "0.0.0.0"` e `"ApiServerPort": 41451`

## 3. Configure the Windows firewall

### Option A: via PowerShell (as Administrator)
```powershell
# Abra PowerShell como Administrador e execute:
New-NetFirewallRule -DisplayName "AirSim API" -Direction Inbound -Protocol TCP -LocalPort 41451 -Action Allow
```

### Option B: via the graphical interface
1. Abra "Windows Defender Firewall com Segurança Avançada"
2. Clique em "Regras de Entrada" → "Nova Regra"
3. Escolha "Porta" → TCP → Porta específica: 41451
4. Permitir a conexão
5. Aplicar a todos os perfis
6. Nome: "AirSim API"

## 4. Find the WSL IP address

No PowerShell do Windows:
```powershell
wsl hostname -I
```

## 5. Reinicie o AirSim

1. Feche completamente o simulador
2. Abra novamente
3. Aguarde carregar completamente

## 6. Test from WSL

Execute no WSL:
```bash
python3 test_quick.py
```

## Troubleshooting

### If it still does not work:

1. **Verifique se a porta está escutando** (no Windows):
```powershell
netstat -an | findstr 41451
```

2. **Desabilite temporariamente o Windows Defender** (para teste)

3. **Use o IP correto do host Windows** (no WSL):
```bash
ip route | grep default | awk '{print $3}'
```

4. **Verifique os logs do AirSim** no Windows

5. **Tente com ApiServerPort diferente** (ex: 9000)