@echo off
REM Script batch pour ajouter Blender au PATH système
REM Exécuter en tant qu'administrateur

set "BLENDER_PATH=C:\Program Files\Blender Foundation\Blender 4.5"

echo 🔍 Vérification de l'installation de Blender...
if exist "%BLENDER_PATH%\blender.exe" (
    echo ✅ Blender trouvé dans : %BLENDER_PATH%
    
    REM Ajouter au PATH système
    echo 📝 Ajout de Blender au PATH système...
    setx PATH "%PATH%;%BLENDER_PATH%" /M
    
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Blender ajouté au PATH système avec succès
        echo ⚠️  Redémarrez votre terminal pour que les changements prennent effet
    ) else (
        echo ❌ Erreur lors de l'ajout au PATH. Vérifiez les permissions d'administrateur.
    )
) else (
    echo ❌ Blender non trouvé dans : %BLENDER_PATH%
    echo Vérifiez le chemin d'installation de Blender
)

echo.
echo 🧪 Test de la commande blender...
blender --version 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ Blender accessible via la ligne de commande
) else (
    echo ❌ Blender non accessible. Redémarrez votre terminal.
)

pause
