import React from 'react';

interface ErrorBannerProps {
  errors: string[];
}

export function ErrorBanner({ errors }: ErrorBannerProps) {
  if (errors.length === 0) {
    return null;
  }

  return (
    <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
      <p className="text-yellow-800 text-sm">
        {errors.join(" • ")}
      </p>
    </div>
  );
}