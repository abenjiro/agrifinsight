# AgriFinSight Web Frontend

A modern React web application for the AgriFinSight smart agriculture platform.

## Features

- **Disease Detection**: Upload plant images for AI-powered disease analysis
- **Crop Analysis**: Get detailed insights about crop health and growth
- **Smart Recommendations**: Receive personalized farming advice
- **Weather Integration**: Weather forecasts and climate-based recommendations
- **Farm Management**: Track multiple farms and their data
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **React Router** for navigation
- **Axios** for API calls
- **Lucide React** for icons

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create environment file:
```bash
cp env.example .env
```

3. Update the API URL in `.env` if needed:
```
VITE_API_URL=http://localhost:8000/api
```

### Development

Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Building for Production

Build the app for production:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

### Linting

Run ESLint:
```bash
npm run lint
```

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Header.tsx      # Navigation header
│   ├── Layout.tsx      # Main layout wrapper
│   └── Sidebar.tsx     # Navigation sidebar
├── pages/              # Page components
│   ├── HomePage.tsx    # Landing page
│   ├── DashboardPage.tsx
│   ├── AnalysisPage.tsx
│   ├── RecommendationsPage.tsx
│   ├── LoginPage.tsx
│   └── RegisterPage.tsx
├── services/           # API services
│   └── api.ts          # API client and services
├── types/              # TypeScript type definitions
│   └── index.ts
├── utils/              # Utility functions
│   └── cn.ts           # Class name utility
├── App.tsx             # Main app component
├── main.tsx            # App entry point
└── index.css           # Global styles
```

## API Integration

The frontend integrates with the AgriFinSight backend API. Key services include:

- **Authentication**: Login, register, user management
- **Farms**: CRUD operations for farm data
- **Analysis**: Image upload and disease detection
- **Recommendations**: AI-powered farming advice

## Styling

The app uses Tailwind CSS with a custom design system:

- **Primary Colors**: Blue theme for main actions
- **Secondary Colors**: Green theme for agriculture-related elements
- **Components**: Reusable button, card, and input styles
- **Responsive**: Mobile-first design approach

## Contributing

1. Follow the existing code style
2. Use TypeScript for all new components
3. Add proper error handling
4. Write meaningful commit messages
5. Test your changes thoroughly




