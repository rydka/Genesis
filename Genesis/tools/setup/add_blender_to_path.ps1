# Script PowerShell pour ajouter Blender au PATH systeme
# Executer en tant qu'administrateur

$blenderPath = "C:\Program Files\Blender Foundation\Blender 4.5"

# Verifier si Blender existe
if (Test-Path $blenderPath) {
    Write-Host "Blender trouve dans : $blenderPath" -ForegroundColor Green

    # Obtenir le PATH actuel
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

    # Verifier si Blender est deja dans le PATH
    if ($currentPath -like "*$blenderPath*") {
        Write-Host "Blender est deja dans le PATH systeme" -ForegroundColor Yellow
    } else {
        # Ajouter Blender au PATH
        $newPath = $currentPath + ";" + $blenderPath
        [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
        Write-Host "Blender ajoute au PATH systeme" -ForegroundColor Green
        Write-Host "Redemarrez votre terminal pour que les changements prennent effet" -ForegroundColor Yellow
    }
} else {
    Write-Host "Blender non trouve dans : $blenderPath" -ForegroundColor Red
    Write-Host "Verifiez le chemin d'installation de Blender" -ForegroundColor Yellow
}

# Tester la commande blender
Write-Host ""
Write-Host "Test de la commande blender..."
try {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    & blender --version
    Write-Host "Blender accessible via la ligne de commande" -ForegroundColor Green
} catch {
    Write-Host "Blender non accessible. Redemarrez votre terminal." -ForegroundColor Red
}
