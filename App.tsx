import React, {useState} from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  useColorScheme,
} from 'react-native';

function App(): React.JSX.Element {
  const isDarkMode = useColorScheme() === 'dark';
  const [count, setCount] = useState(0);

  const backgroundStyle = {
    backgroundColor: isDarkMode ? '#1a1a1a' : '#f5f5f5',
    flex: 1,
  };

  return (
    <SafeAreaView style={backgroundStyle}>
      <StatusBar
        barStyle={isDarkMode ? 'light-content' : 'dark-content'}
        backgroundColor={backgroundStyle.backgroundColor}
      />
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        style={backgroundStyle}
        contentContainerStyle={styles.scrollViewContent}>
        <View style={styles.container}>
          <View style={styles.header}>
            <Text style={[styles.title, {color: isDarkMode ? '#fff' : '#000'}]}>
              Welcome to ML App
            </Text>
            <Text style={[styles.subtitle, {color: isDarkMode ? '#aaa' : '#666'}]}>
              React Native for iOS
            </Text>
          </View>

          <View style={styles.card}>
            <Text style={[styles.cardTitle, {color: isDarkMode ? '#fff' : '#000'}]}>
              Counter Demo
            </Text>
            <Text style={[styles.countText, {color: isDarkMode ? '#4CAF50' : '#2196F3'}]}>
              {count}
            </Text>
            <View style={styles.buttonRow}>
              <TouchableOpacity
                style={[styles.button, styles.decrementButton]}
                onPress={() => setCount(count - 1)}>
                <Text style={styles.buttonText}>-</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.button, styles.resetButton]}
                onPress={() => setCount(0)}>
                <Text style={styles.buttonText}>Reset</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.button, styles.incrementButton]}
                onPress={() => setCount(count + 1)}>
                <Text style={styles.buttonText}>+</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.infoCard}>
            <Text style={[styles.infoTitle, {color: isDarkMode ? '#fff' : '#000'}]}>
              Features
            </Text>
            <View style={styles.featureList}>
              <Text style={[styles.featureItem, {color: isDarkMode ? '#ddd' : '#333'}]}>
                ✓ TypeScript support
              </Text>
              <Text style={[styles.featureItem, {color: isDarkMode ? '#ddd' : '#333'}]}>
                ✓ Dark mode support
              </Text>
              <Text style={[styles.featureItem, {color: isDarkMode ? '#ddd' : '#333'}]}>
                ✓ iOS optimized
              </Text>
              <Text style={[styles.featureItem, {color: isDarkMode ? '#ddd' : '#333'}]}>
                ✓ Hot reload enabled
              </Text>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  scrollViewContent: {
    flexGrow: 1,
  },
  container: {
    flex: 1,
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginTop: 40,
    marginBottom: 40,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 18,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  cardTitle: {
    fontSize: 24,
    fontWeight: '600',
    marginBottom: 20,
    textAlign: 'center',
  },
  countText: {
    fontSize: 72,
    fontWeight: 'bold',
    textAlign: 'center',
    marginVertical: 20,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  button: {
    flex: 1,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  incrementButton: {
    backgroundColor: '#4CAF50',
  },
  decrementButton: {
    backgroundColor: '#f44336',
  },
  resetButton: {
    backgroundColor: '#2196F3',
  },
  buttonText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '600',
  },
  infoCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  infoTitle: {
    fontSize: 22,
    fontWeight: '600',
    marginBottom: 16,
  },
  featureList: {
    gap: 12,
  },
  featureItem: {
    fontSize: 16,
    lineHeight: 24,
  },
});

export default App;
