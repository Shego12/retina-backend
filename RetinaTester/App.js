import React, { useState, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isScanning, setIsScanning] = useState(false);
  const [flashColor, setFlashColor] = useState('transparent');
  const cameraRef = useRef(null);

  if (!permission) return <View style={styles.container} />;
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>Retina requires camera access</Text>
        <TouchableOpacity onPress={requestPermission} style={styles.authButton}>
          <Text style={styles.authButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const triggerLivenessSequence = async () => {
    const colors = ['rgba(255,0,0,0.4)', 'rgba(0,255,0,0.4)', 'rgba(0,0,255,0.4)'];
    
    for (let color of colors) {
      setFlashColor(color);
      await new Promise(resolve => setTimeout(resolve, 150)); 
    }
    setFlashColor('transparent');
  };

  const handleRecognize = async () => {
    if (cameraRef.current && !isScanning) {
      setIsScanning(true); 
      
      await triggerLivenessSequence();
      
      try {
        const photo = await cameraRef.current.takePictureAsync({ quality: 0.5 });
        
        const formData = new FormData();
        formData.append('file', {
          uri: photo.uri,
          type: 'image/jpeg',
          name: 'checkin.jpg',
        });

        // UPDATED TO RAILWAY URL
        const response = await fetch('https://retina-backend-production.up.railway.app/recognize', {
          method: 'POST',
          body: formData,
          headers: { 'Content-Type': 'multipart/form-data' },
        });

        const result = await response.json();
        
        if (result.status === "success") {
          Alert.alert("Access Granted", `Welcome, ${result.message.replace('Welcome, ', '').replace('!', '')}`);
        } else {
          Alert.alert("Access Denied", "Face not recognized in the system.");
        }
      } catch (err) {
        Alert.alert("System Error", "Could not connect to the Retina servers.");
      } finally {
        setIsScanning(false); 
      }
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.logo}>RETINA</Text>
        <Text style={styles.subtitle}>Identity Verification System</Text>
      </View>

      <View style={styles.cameraWrapper}>
        <CameraView style={styles.camera} facing="front" ref={cameraRef}>
          <View style={[StyleSheet.absoluteFillObject, { backgroundColor: flashColor }]} />
          <View style={styles.reticleContainer}>
            <View style={[styles.corner, styles.topLeft]} />
            <View style={[styles.corner, styles.topRight]} />
            <View style={[styles.corner, styles.bottomLeft]} />
            <View style={[styles.corner, styles.bottomRight]} />
          </View>
        </CameraView>
      </View>

      <View style={styles.footer}>
        <Text style={styles.instructionText}>
          {isScanning ? "Analyzing liveness and biometrics..." : "Align face within the frame"}
        </Text>
        
        <TouchableOpacity 
          style={[styles.authButton, isScanning && styles.authButtonDisabled]} 
          onPress={handleRecognize}
          disabled={isScanning}
        >
          {isScanning ? (
            <ActivityIndicator color="#000" />
          ) : (
            <Text style={styles.authButtonText}>AUTHENTICATE</Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0a0a', justifyContent: 'space-between', paddingTop: 60, paddingBottom: 40 },
  header: { alignItems: 'center', marginBottom: 20 },
  logo: { fontSize: 32, fontWeight: '900', color: '#00D2FF', letterSpacing: 4 },
  subtitle: { fontSize: 12, color: '#888', letterSpacing: 1, textTransform: 'uppercase', marginTop: 4 },
  cameraWrapper: { height: 450, marginHorizontal: 20, borderRadius: 20, overflow: 'hidden', borderWidth: 2, borderColor: '#222' },
  camera: { flex: 1 },
  reticleContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 30 },
  corner: { position: 'absolute', width: 40, height: 40, borderColor: '#00D2FF' },
  topLeft: { top: 60, left: 40, borderTopWidth: 4, borderLeftWidth: 4 },
  topRight: { top: 60, right: 40, borderTopWidth: 4, borderRightWidth: 4 },
  bottomLeft: { bottom: 60, left: 40, borderBottomWidth: 4, borderLeftWidth: 4 },
  bottomRight: { bottom: 60, right: 40, borderBottomWidth: 4, borderRightWidth: 4 },
  footer: { alignItems: 'center', paddingHorizontal: 20 },
  instructionText: { color: '#aaa', marginBottom: 20, fontSize: 14 },
  authButton: { backgroundColor: '#00D2FF', width: '100%', padding: 18, borderRadius: 12, alignItems: 'center', shadowColor: '#00D2FF', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 10, elevation: 8 },
  authButtonDisabled: { backgroundColor: '#555', shadowOpacity: 0 },
  authButtonText: { fontSize: 16, fontWeight: 'bold', color: '#000', letterSpacing: 2 },
  errorText: { color: 'white', textAlign: 'center', marginBottom: 20, fontSize: 16 }
});