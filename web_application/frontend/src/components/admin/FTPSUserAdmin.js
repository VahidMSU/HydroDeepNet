import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';

const FTPSUserAdmin = () => {
  const [username, setUsername] = useState('');
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [credentials, setCredentials] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedUser, setSelectedUser] = useState('');

  // Fetch user list on component mount
  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/ftps/list', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });

      const data = await response.json();
      if (data.success) {
        setUsers(data.users || []);
      } else {
        setError(data.message || 'Failed to fetch FTPS users');
      }
    } catch (err) {
      setError('Error connecting to server');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const createUser = async () => {
    if (!username) {
      setError('Username is required');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);
    setCredentials(null);

    try {
      const response = await fetch('/api/ftps/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ username }),
      });

      const data = await response.json();
      if (data.success) {
        setSuccess(`FTPS user ${username} created successfully`);
        setCredentials({
          username: data.username,
          password: data.password,
          server: data.server,
          port: data.port,
          protocol: data.protocol,
        });
        setUsername('');
        fetchUsers();
      } else {
        setError(data.message || 'Failed to create FTPS user');
      }
    } catch (err) {
      setError('Error connecting to server');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const confirmDelete = (user) => {
    setSelectedUser(user);
    setOpenDialog(true);
  };

  const deleteUser = async () => {
    setOpenDialog(false);
    if (!selectedUser) return;

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch('/api/ftps/delete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ username: selectedUser }),
      });

      const data = await response.json();
      if (data.success) {
        setSuccess(`FTPS user ${selectedUser} deleted successfully`);
        fetchUsers();
      } else {
        setError(data.message || 'Failed to delete FTPS user');
      }
    } catch (err) {
      setError('Error connecting to server');
      console.error(err);
    } finally {
      setLoading(false);
      setSelectedUser('');
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4 }}>
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" gutterBottom>
          FTPS User Management
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          Create and manage FTPS users for secure data access. Users will have read-only access to
          SWATplus model data.
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            {success}
          </Alert>
        )}

        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <TextField
            label="Username"
            variant="outlined"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            fullWidth
            size="small"
          />
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={createUser}
            disabled={loading}
          >
            Create User
          </Button>
        </Box>

        {credentials && (
          <Paper elevation={1} sx={{ p: 2, mb: 3, bgcolor: '#f8f9fa' }}>
            <Typography variant="h6" gutterBottom>
              New User Credentials
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Please save these credentials. The password will not be shown again.
            </Typography>
            <Box
              component="pre"
              sx={{ p: 2, bgcolor: '#e9ecef', borderRadius: 1, overflow: 'auto' }}
            >
              {`Username: ${credentials.username}
Password: ${credentials.password}
Server: ${credentials.server}
Port: ${credentials.port}
Protocol: ${credentials.protocol}`}
            </Box>
          </Paper>
        )}
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">FTPS Users</Typography>
          <Button startIcon={<RefreshIcon />} size="small" onClick={fetchUsers} disabled={loading}>
            Refresh
          </Button>
        </Box>

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : users.length > 0 ? (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Username</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {users.map((user) => (
                  <TableRow key={user}>
                    <TableCell>{user}</TableCell>
                    <TableCell align="right">
                      <Button
                        startIcon={<DeleteIcon />}
                        color="error"
                        size="small"
                        onClick={() => confirmDelete(user)}
                      >
                        Delete
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ p: 2, textAlign: 'center' }}>
            No FTPS users found
          </Typography>
        )}
      </Paper>

      {/* Confirmation Dialog */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false)}>
        <DialogTitle>Confirm Deletion</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete FTPS user "{selectedUser}"? This action cannot be
            undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button onClick={deleteUser} color="error" autoFocus>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default FTPSUserAdmin;
