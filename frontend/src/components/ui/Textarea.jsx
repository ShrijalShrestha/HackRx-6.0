import React from "react";

export const Textarea = ({ className = "", ...props }) => {
  return (
    <textarea
      className={`border border-gray-300 rounded px-3 py-2 w-full focus:outline-none focus:ring-2 focus:ring-indigo-500 ${className}`}
      rows="4"
      {...props}
    />
  );
};
