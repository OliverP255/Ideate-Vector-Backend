# Frontend Service

Next.js-based frontend for the Knowledge Map visualization.

## Features

- Interactive map visualization using deck.gl
- Document exploration and retrieval
- User overlay showing consumed resources
- Responsive design with TailwindCSS
- Real-time updates

## Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Run tests
npm test
```

## Technology Stack

- Next.js 14 (React framework)
- TypeScript
- deck.gl (WebGL visualization)
- TailwindCSS (styling)
- React Query (data fetching)

## Project Structure

- `app/` - Next.js app directory
- `pages/` - Legacy pages (if needed)
- `components/` - React components
- `public/` - Static assets
- `styles/` - CSS and styling files

## API Integration

The frontend communicates with the backend API at `/api/*` endpoints for:
- Map data retrieval
- Document search and click handling
- User overlay management
- Health monitoring
