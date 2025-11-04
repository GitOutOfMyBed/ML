# ML App - React Native iOS Application

A React Native application built specifically for iOS with TypeScript support.

## Features

- TypeScript support for type-safe development
- Dark mode support
- iOS-optimized UI components
- Hot reload for fast development
- Modern React Native architecture
- Interactive counter demo
- Beautiful, responsive design

## Prerequisites

Before you begin, ensure you have the following installed:

- Node.js (>= 18)
- Xcode (>= 13.0) - [Download from Mac App Store](https://apps.apple.com/us/app/xcode/id497799835)
- CocoaPods - Install with: `sudo gem install cocoapods`
- React Native CLI - Install with: `npm install -g react-native-cli`

## Installation

1. Install dependencies:
```bash
npm install
```

2. Install iOS pods:
```bash
cd ios && pod install && cd ..
```

## Running the App

### Development Mode

1. Start the Metro bundler:
```bash
npm start
```

2. In a new terminal, run the iOS app:
```bash
npm run ios
```

Or run on a specific simulator:
```bash
npm run ios -- --simulator="iPhone 15 Pro"
```

### Available Scripts

- `npm start` - Start the Metro bundler
- `npm run ios` - Run the app on iOS simulator
- `npm test` - Run tests
- `npm run lint` - Run ESLint

## Project Structure

```
ML/
├── App.tsx              # Main application component
├── index.js             # Application entry point
├── package.json         # Dependencies and scripts
├── tsconfig.json        # TypeScript configuration
├── babel.config.js      # Babel configuration
├── metro.config.js      # Metro bundler configuration
├── ios/                 # iOS native code
│   ├── MLApp/          # iOS app files
│   ├── Podfile         # CocoaPods dependencies
│   └── MLApp.xcodeproj # Xcode project
└── sgd_image_changer.py # Existing Python script
```

## App Features

### Counter Demo
The app includes an interactive counter with:
- Increment button (+)
- Decrement button (-)
- Reset button
- Real-time state updates

### Dark Mode
Automatically detects and responds to system dark mode preferences.

## Development

### Debugging

- Press `Cmd + D` in the iOS simulator to open the developer menu
- Enable "Debug JS Remotely" for debugging in Chrome DevTools
- Use "Fast Refresh" for instant UI updates

### Building for Production

To create a production build:

1. Open `ios/MLApp.xcworkspace` in Xcode
2. Select "Generic iOS Device" or your connected device
3. Go to Product > Archive
4. Follow the App Store submission process

## Troubleshooting

### Metro bundler not starting
```bash
npm start -- --reset-cache
```

### iOS build fails
```bash
cd ios
pod deintegrate
pod install
cd ..
```

### Clear all caches
```bash
watchman watch-del-all
rm -rf node_modules
npm install
cd ios && pod install && cd ..
```

## Requirements

- iOS 13.4 or later
- iPhone or iPad device/simulator

## License

This project is private and proprietary.

## Support

For issues or questions, please refer to the [React Native documentation](https://reactnative.dev/).
